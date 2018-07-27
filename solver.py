import time
import torch
from torchvision.utils import save_image
from models import modelset, TripletLoss, SDLLoss
from datasets import *
import torch.nn.functional as F
import os

class Solver(object):

    def __init__(self, config):
        self.config = config

        # construct data loader
        if config.obj.lower() in ['shoes', 'handbags']:
            self.dataset = Pix2PixData(config.data_root, obj=config.obj, image_size=config.image_size, random=config.random)            
        elif config.obj.lower() == 'mnist':
            self.dataset = MnistSvhnData(config.data_root, random=config.random)
        self.data_loader = self.dataset.get_loader(batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

        print(len(self.dataset))

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            if self.config.image_size == 64 or config.obj=='mnist':
                self.device2 = self.device
            else:
                self.device2 = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
            exit(0)
        self.cpu_device = torch.device("cpu")

        self.triplet_loss = TripletLoss(delta=self.config.margin)

        self.build()

        self.has_logged = False

    def build(self):

        flag = 'mnist' if self.config.obj == 'mnist' else 'pix2pix'

        # networks
        self.E = modelset[flag]['encoder'](self.config.conv_dim, image_size=self.config.image_size)
        self.G = modelset[flag]['decoder'](self.config.conv_dim, image_size=self.config.image_size)
        self.D = modelset[flag]['discriminator'](self.config.conv_dim, image_size=self.config.image_size)
        self.E.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device2)

        # optimizor
        self.opt_E = torch.optim.Adam(self.E.parameters(), lr=self.config.g_lr, betas=(self.config.beta1, self.config.beta2))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config.g_lr, betas=(self.config.beta1, self.config.beta2))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config.d_lr, betas=(self.config.beta1, self.config.beta2))

        # inverse tensor
        self.inv_idx = torch.arange(self.config.batch_size-1, -1, -1).long().to(self.device)
        if self.config.obj == 'mnist':
            self.sdl = SDLLoss(self.config.conv_dim, self.config.conv_dim*15, self.config.sdl_alpha)
            self.sdl.to(device)


    def reset_grad(self):
        self.opt_E.zero_grad()
        self.opt_G.zero_grad()
        self.opt_D.zero_grad()

    def update_lr(self, g_lr, d_lr):
        for param_group in self.opt_E.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.opt_G.param_groups:
            param_group['lr'] = g_lr           
        for param_group in self.opt_D.param_groups:
            param_group['lr'] = d_lr

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def load_pretrained_model(self):
        e = eval(self.config.pretrained_model)
        E_path = os.path.join(self.config.model_save_path, '{}-E.ckpt'.format(e))
        G_path = os.path.join(self.config.model_save_path, '{}-G.ckpt'.format(e))
        D_path = os.path.join(self.config.model_save_path, '{}-D.ckpt'.format(e))
        self.E.load_model(E_path)
        self.G.load_model(G_path)
        self.D.load_model(D_path)
        print('Load model checkpoints from {}...'.format(self.config.model_save_path))

    
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device2)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)    

    def rec_loss(self, x1, x2, mode='pho'):
        assert mode in ['pho', 'skt']
        n = x1.size(0)
        if mode == 'pho' and self.config.ablation_recon_kl:
            return torch.mean(torch.abs(x1-x2))
        elif mode == 'skt':
            eps = 1e-10
            loss = (x1-x2) * (torch.log(x1 + eps)-torch.log(x2 + eps))
            #loss += (x1-x2).abs().mean()
            return loss.mean()

    def cons_loss(self, z1, z2):
        l2_loss = (z1-z2).pow(2).mean()
        if not self.config.ablation_const_l2:
            l2_loss = l2_loss.detach()
        angle = torch.sum(z1*z2, dim=1) / (torch.norm(z1, p=2, dim=1) * torch.norm(z2, p=2, dim=1))
        angle_loss = 1 - (angle).mean()
        if not self.config.ablation_const_angle:
            angle_loss = angle_loss.detach()
        return l2_loss,  angle_loss

    def shuffle(self, x):
        rand_idx = torch.randperm(x[0].size(0)).to(self.device)
        return [x[0][rand_idx], x[1][rand_idx]]


    def mix(self, feat_pho, feat_skt, shuffle=False, forg=False):
    
        if shuffle:
            feat_pho_shf = self.shuffle(feat_pho)
            feat_skt_shf = self.shuffle(feat_skt)
        else:
            feat_pho_shf = feat_pho
            feat_skt_shf = feat_skt            
        
        if not forg:
            feat_skt_pho = [feat_skt[0], feat_pho_shf[1]]
            feat_pho_skt = [feat_pho[0], feat_skt_shf[1]]
        else:
            feat_skt_pho = [feat_skt_shf[0], feat_pho[1]]
            feat_pho_skt = [feat_pho_shf[0], feat_skt[1]]

        return feat_skt_pho, feat_pho_skt

    def train_AE(self, skts, phos, loss, mutual=True):
        ## fake samples
        feat_pho = self.E(phos, 'pho', 'map')
        feat_skt = self.E(skts, 'skt', 'map')  
        
        ## corresponded images
        if mutual:
            feat_skt_pho, feat_pho_skt = self.mix(feat_pho, feat_skt, shuffle=False)
            rec_phos = self.G(feat_skt_pho, 'pho')
            rec_skts = self.G(feat_pho_skt, 'skt')       
        else:
            rec_phos = self.G(feat_pho, 'pho')
            rec_skts = self.G(feat_skt, 'skt')                       

        loss_rec_pho = self.rec_loss(phos, rec_phos, 'pho')
        loss_rec_skt = self.rec_loss(skts, rec_skts, 'skt')
        l2_loss, angle_loss = self.cons_loss(feat_pho[0],feat_skt[0])
        
        rec_loss = self.config.lambda_rec * (loss_rec_pho+loss_rec_skt)\
                   + self.config.lambda_constrain * (l2_loss+angle_loss)

        self.reset_grad()
        rec_loss.backward()
        self.opt_G.step()                    
        self.opt_E.step() 

        loss['L_recon[pho]'] = loss_rec_pho.item()
        loss['L_recon[skt]'] = loss_rec_skt.item()
        loss['L_const(l2)'] = l2_loss.item()
        loss['L_const(angle)'] = angle_loss.item()

    def train_IR(self, skts, phos, loss):

        # using photos in the same batch as neg samples
        phos_pos = phos
        #phos_neg = phos[self.inv_idx]

        feat_ppho = self.E(phos_pos, 'pho', 'tpl')
        #feat_npho = self.E(phos_neg, 'pho', 'tpl')
        feat_skt = self.E(skts, 'skt', 'tpl')

        losses = {}
        losses['max'], losses['mean'] = self.triplet_loss(feat_skt[0], feat_ppho[0], 0)
        e_loss = sum([losses[f] for f in self.config.ablation_tri]) * self.config.lambda_triplet

        self.reset_grad()
        e_loss.backward()
        self.opt_E.step() 

        loss['L_tri(max)'] = losses['max'].item()
        loss['L_tri(mean)'] = losses['mean'].item()

    def train_SDL(self, skts, phos, loss):

        # using photos in the same batch as neg samples
        phos_pos = phos
        #phos_neg = phos[self.inv_idx]

        feat_ppho = self.E(phos_pos, 'pho', 'tpl')[0]
        #feat_npho = self.E(phos_neg, 'pho', 'tpl')
        feat_skt = self.E(skts, 'skt', 'tpl')[0]

        sdl_loss = self.sdl(feat_skt, feat_ppho)
        e_loss = sdl_loss * self.config.lambda_triplet

        self.reset_grad()
        e_loss.backward()
        self.opt_E.step() 

        loss['L_tri(max)'] = sdl_loss.item()
        loss['L_tri(mean)'] = 0


    def train_dis(self, skts, phos, loss):

        # generate fake images
        feat_pho = self.E(phos, 'pho', 'map')
        feat_skt = self.E(skts, 'skt', 'map')   

        feat_skt_pho, feat_pho_skt = self.mix(feat_pho, feat_skt, shuffle=True)     
        fake_phos = self.G(feat_skt_pho, 'pho')
        fake_skts = self.G(feat_pho_skt, 'skt') 

        # move to gpu2
        skts = skts.to(self.device2).detach()
        phos = phos.to(self.device2).detach()
        fake_skts = fake_skts.to(self.device2).detach()
        fake_phos = fake_phos.to(self.device2).detach()

        # train D
        real_dis_pho = self.D(phos, 'pho').mean()
        real_dis_skt = self.D(skts, 'skt').mean()
        fake_dis_pho = self.D(fake_phos, 'pho').mean()
        fake_dis_skt = self.D(fake_skts, 'skt').mean()

        # gradient panelty
        alpha = torch.rand(phos.size(0), 1, 1, 1).to(self.device2)
        phos_hat = (alpha * phos.data + (1 - alpha) * fake_phos.data).requires_grad_(True)
        out_src_pho = self.D(phos_hat, 'pho')
        gp_pho = self.gradient_penalty(out_src_pho, phos_hat)        
        
        alpha = torch.rand(skts.size(0), 1, 1, 1).to(self.device2)
        skts_hat = (alpha * skts.data + (1 - alpha) * fake_skts.data).requires_grad_(True)
        out_src_skt = self.D(skts_hat, 'skt')
        gp_skt = self.gradient_penalty(out_src_skt, skts_hat)   
        
        d_loss = -(real_dis_pho+real_dis_skt) + (fake_dis_pho+fake_dis_skt)+self.config.lambda_gp * (gp_pho+gp_skt)
        self.reset_grad()
        d_loss.backward()
        self.opt_D.step()  

        # logging  
        loss['L_gan(dis/real)[pho]'] = real_dis_pho.item()
        loss['L_gan(dis/real)[skt]'] = real_dis_skt.item()
        loss['L_gan(dis/fake)[pho]'] = fake_dis_pho.item()
        loss['L_gan(dis/fake)[skt]'] = fake_dis_skt.item()
        loss['L_gan(gp)[pho]'] = gp_pho.item()
        loss['L_gan(gp)[skt]'] = gp_skt.item()

    def train_gen(self, skts, phos, loss, triplet=False):
        
        # generate fake images
        feat_pho = self.E(phos, 'pho', 'map')
        feat_skt = self.E(skts, 'skt', 'map')   

        feat_skt_pho, feat_pho_skt = self.mix(feat_pho, feat_skt, shuffle=True, forg=True)     
        if triplet:
            feat_skt_pho = [feat_skt_pho[0].detach(), feat_skt_pho[1].detach()]
            feat_pho_skt = [feat_pho_skt[0].detach(), feat_pho_skt[1].detach()]
        fake_phos = self.G(feat_skt_pho, 'pho')
        fake_skts = self.G(feat_pho_skt, 'skt') 

        if triplet:
            # pass through E
            vec_pho = F.normalize(self.E(fake_phos, 'pho', 'tpl'))   
            vec_skt = F.normalize(self.E(fake_skts, 'skt', 'tpl'))  

            triplet_loss = F.normalize(vec_pho-vec_skt)

        # pass through D
        fake_phos = fake_phos.to(self.device2)
        fake_skts = fake_skts.to(self.device2)
        fake_dis_pho = torch.mean(self.D(fake_phos, 'pho'))
        fake_dis_skt = torch.mean(self.D(fake_skts, 'skt'))  

        ## construct loss
        gen_loss = -(fake_dis_pho+fake_dis_skt).to(self.device) 
        if triplet:
            gen_loss += self.lambda_triplet * triplet_loss        

        self.reset_grad()
        gen_loss.backward()
        self.opt_G.step()  
        if not triplet:                  
            self.opt_E.step()   

        # log
        loss['L_gan(gen/fake)[pho]'] = fake_dis_pho.item()
        loss['L_gan(gen/fake)[skt]'] = fake_dis_skt.item()
        if triplet:  
            loss['AE/loss_triplet_gen'] = triplet_loss.item()


    def train(self):
        start_time = time.time()
        num_epochs = self.config.num_epochs
        num_epochs_decay = self.config.num_epochs_decay
        num_iters = self.config.num_iters
        g_lr = self.config.g_lr
        d_lr = self.config.d_lr
        

        if self.config.pretrained_model is None:
            start_epoch = 0
            if os.path.exists(self.config.log_file):
                os.remove(self.config.log_file)
        else:
            self.load_pretrained_model()
            start_epoch = eval(self.config.pretrained_model)

        for e in range(start_epoch, num_epochs):
            for i in range(num_iters):

                # sample_data
                try:
                    skts, phos = next(pair_iter)
                    if skts.size(0) == 1:
                        raise ValueError
                except:
                    pair_iter = iter(self.data_loader)
                    skts, phos = next(pair_iter)            

                phos = phos.to(self.device)
                skts = skts.to(self.device)
                loss = {}
            

                # train auto-encoder
                #self.train_AE(skts, phos, loss, mutual=True)

                # train discriminator
                self.train_dis(skts, phos, loss)

                #self.train_IR(skts, phos, loss)

                if (i+1) % self.config.d_train_repeat == 0:
                    self.train_AE(skts, phos, loss, mutual=True)
                    if self.config.obj == 'mnist':
                        self.train_SDL(skts, phos, loss)
                    else:
                        self.train_IR(skts, phos, loss)
                    self.train_gen(skts, phos, loss)

                
                # Print out training information.
                if (i+1) % self.config.log_step == 0:
                    et = time.time() - start_time
                    et = str(et)[:-12]
                    log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}/{}]\n".format(et, e+1, num_epochs, i+1, num_iters)
                    log += 'L_recon: ' + '[pho]:{:.4f}, [skt]:{:.4f}\n'.format(loss['L_recon[pho]'], loss['L_recon[skt]'])
                    log += 'L_const: ' + '[l2]:{:.4f}, [angle]:{:.4f}\n'.format(loss['L_const(l2)'], loss['L_const(angle)'])
                    log += 'L_tri: ' + '[max]:{:.4f}, [mean]:{:.4f}\n'.format(loss['L_tri(max)'], loss['L_tri(mean)'])
                    log += 'L_gan [pho]: ' + '(dis/real):{:.4f}, (dis/fake):{:.4f}, (gen/fake):{:.4f}, (gp):{:.4f}\n'.format(
                        loss['L_gan(dis/real)[pho]'], loss['L_gan(dis/fake)[pho]'], loss['L_gan(gen/fake)[pho]'],  loss['L_gan(gp)[pho]'])
                    log += 'L_gan [skt]: ' + '(dis/real):{:.4f}, (dis/fake):{:.4f}, (gen/fake):{:.4f}, (gp):{:.4f}\n'.format(
                        loss['L_gan(dis/real)[skt]'], loss['L_gan(dis/fake)[skt]'], loss['L_gan(gen/fake)[skt]'],  loss['L_gan(gp)[skt]'])

                    print(log)  

                    # write loss into log file
                    with open(self.config.log_file, 'a+') as f:
                        if not self.has_logged:
                            f.write('epoch')
                            f.write(';')
                            for key in loss.keys():
                                f.write(key)
                                f.write(';')
                            f.write('\n')
                            self.haslogged = True

                        f.write(str(e+1))
                        f.write(';')
                        for l in loss.values():
                            f.write('{:.4f}'.format(l))
                            f.write(';')
                

            # Translate fixed images for debugging.
            sample_path = os.path.join(self.config.sample_path, '{}_{}-images.jpg'.format(e+1, i+1))
            self.log_image(skts, phos, sample_path)
             
            # Save model checkpoints.
            if (e+1) % self.config.model_save_step == 0:
                E_path = os.path.join(self.config.model_save_path, '{}-E.ckpt'.format(e+1))
                G_path = os.path.join(self.config.model_save_path, '{}-G.ckpt'.format(e+1))
                D_path = os.path.join(self.config.model_save_path, '{}-D.ckpt'.format(e+1))
                self.E.save_model(E_path)
                self.G.save_model(G_path)
                self.D.save_model(D_path)
                print('Saved model checkpoints into {}...'.format(self.config.model_save_path))


            # Decay learning rates.
            if (e+1) > (num_epochs - num_epochs_decay):
                g_lr -= (self.config.g_lr / float(num_epochs_decay))
                d_lr -= (self.config.d_lr / float(num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))            



    def log_image(self, skts, phos, sample_path):

        inv_idx = torch.arange(phos.size(0)-1, -1, -1).long()


        pho_fake_list = [phos.cpu()]
        skt_fake_list = [skts.cpu()[inv_idx]]

        with torch.no_grad():
            feat_pho = self.E(phos, 'pho', 'map')
            feat_skt = self.E(skts, 'skt', 'map') 
                                     
            # rec
            ## corresponded images
            feat_skt_pho, feat_pho_skt = self.mix(feat_pho, feat_skt, shuffle=False)
            rec_phos = self.G(feat_skt_pho, 'pho')
            rec_skts = self.G(feat_pho_skt, 'skt')  
            pho_fake_list.append(rec_phos.cpu())
            skt_fake_list.append(rec_skts.cpu()[inv_idx])

            # gen
            feat_skt_pho = [feat_skt_pho[0][inv_idx.to(self.device)], feat_skt_pho[1]]
            fake_phos = self.G(feat_skt_pho, 'pho')
            feat_pho_skt = [feat_pho_skt[0][inv_idx.to(self.device)], feat_pho_skt[1]]
            fake_skts = self.G(feat_pho_skt, 'skt')            
            pho_fake_list.append(fake_phos.cpu())
            skt_fake_list.append(fake_skts.cpu()[inv_idx])

        pho_concat = self.denorm(torch.cat(pho_fake_list, dim=3))
        skt_concat = torch.cat(skt_fake_list[::-1], dim=3)
        #skt_concat = skt_concat.expand(skt_concat.size(0), 3, skt_concat.size(2), skt_concat.size(3))
        x_concat = torch.cat([pho_concat, skt_concat], dim=3)
        save_image(x_concat, sample_path, nrow=1, padding=0)
        save_image(x_concat, os.path.join(self.config.latest_path, 'latest.png'), nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(sample_path))           












