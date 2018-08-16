from itertools import chain
import random
from dataset import *
from option import TrainOption
from loss import *

def main():
    global args
    args = TrainOption().parse()

    print(args.result_path)
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    #[dataset]
    # batch size의 image 불러오기
    test, train = get_real_image(args.image_size, os.path.join(args.input_path), args.test_size)



    e = len(test)

    test = np.hstack([test[0:e-4], train[1:e-3], train[2:e-2], train[3:e-1]])
    test_gt = train[4:e]

    print("e :", e)
    print("test shape :", test.shape)
    print("test_gt shape :", test_gt.shape)

    test = Variable(torch.FloatTensor(test))

    generator = Generator()
    discriminator = Discriminator()

    if torch.cuda:
        test = test.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    #[create model]

    #[training]


    data_size = len(train)
    n_batchs = data_size // args.batch_size
    print(data_size)
    print(args.batch_size)
    print(n_batchs)


    optim_gen = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iter = 0
    for epoch in range(args.epoch):
        print("epoch :", epoch)

        generator.zero_grad()
        discriminator.zero_grad()

        for i in range(n_batchs):
            #A = train[args.batch_size * i:args.batch_size * (i + 1)]

            s = args.batch_size * i
            e = args.batch_size * (i + 1)

            #print("trint size :", len(train), "e : ", e)
            if e + 3 > len(train):
                break
            A = np.hstack([train[s:e], train[s+1:e+1], train[s+2:e+2], train[s+3:e+3]])
            gt = train[s+4:e+4]

            # batch size(8) * chennel(3 * 4) * width * height
            #print("A shape :", A.shape)
            #print("gt shape :", gt.shape)

            A = Variable(torch.FloatTensor(A))
            gt = Variable(torch.FloatTensor(gt))
            if torch.cuda:
                A = A.cuda()
                gt = gt.cuda()

            g_image = generator(A)
            dis_real = discriminator(gt)
            dis_fake = discriminator(g_image)


            if iter == 0:
                print("g_image :", g_image.shape)
                print("gt :", gt.shape)
                print("dis shape :", dis_real.shape)

            mse_loss = get_mse_loss(gt, g_image, nn.MSELoss())
            dis_loss, gen_loss = get_gan_loss(dis_fake, dis_real, nn.BCELoss())

            gen_loss = gen_loss * 0.05 + mse_loss * 1

            # loss
            if iter % args.print_iter == 0:
                print("dis loss : {0:.3f}, get loss : {1:.3f}".format(dis_loss, gen_loss))

            if iter % 3 == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()


            if iter % args.save_iter == 0:
                n_iter = iter // args.save_iter
                print("start to save image and model[", n_iter, "]")
                save_path = os.path.join(args.result_path, str(n_iter))

                save_all_image(save_path, generator, test, test_gt)

            iter = iter + 1


if __name__=="__main__":
    main()