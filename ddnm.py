import torch
import torchvision.utils as tvu
import os

def simplified_ddnm_plus(model, x_orig, y, A, Ap, sigma_y, sigma_t, a_t, gamma_t):

    # to account for scaling to [-1,1]
    sigma_y = 2 * sigma_y

    y = A(x_orig)
    Apy = Ap(y)

    sample_size = model.config.sample_size
    x = torch.randn((y.shape[0], 3, sample_size, sample_size)).to("cuda")

        with torch.no_grad():
            skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
            n = x.size(0)
            x0_preds = []
            xs = [x]

            times = get_schedule_jump(config.time_travel.T_sampling,
                                      config.time_travel.travel_length,
                                      config.time_travel.travel_repeat,
                                      )
            time_pairs = list(zip(times[:-1], times[1:]))

            # reverse diffusion sampling
            for i, j in time_pairs:
                i, j = i * skip, j * skip
                if j < 0: j = -1

                if j < i:  # normal sampling
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at_next ** 2).sqrt()
                    xt = xs[-1].to('cuda')

                    print(f'at_next: {at_next}, sigma_t: {sigma_t}')
                    print(t)

                    et = model(xt, t)

                    if et.size(1) == 6:
                        et = et[:, :3]

                    # Eq. 12
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # Eq. 19
                    if sigma_t >= at_next * sigma_y:
                        lambda_t = 1.
                        gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                    else:
                        lambda_t = (sigma_t) / (at_next * sigma_y)
                        gamma_t = 0.

                    # Eq. 17
                    x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)

                    eta = self.args.eta

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # different from the paper, we use DDIM here instead of DDPM
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                else:  # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t = x0_preds[-1].to('cuda')

                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to('cpu'))

            x = xs[-1]

        x = [inverse_data_transform(config, xi) for xi in x]

        tvu.save_image(
            x[0], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
        )
        orig = inverse_data_transform(config, x_orig[0])
        mse = torch.mean((x[0].to(self.device) - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        avg_psnr += psnr

        idx_so_far += y.shape[0]

        # pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

    avg_psnr = avg_psnr / (idx_so_far - idx_init)
    print("Total Average PSNR: %.2f" % avg_psnr)
    print("Number of samples: %d" % (idx_so_far - idx_init))