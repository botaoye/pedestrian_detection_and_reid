from torchvision import transforms
from torchvision.datasets.folder import default_loader

import os
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')

import torch

from reid.network import MGN
from reid.utils.extract_feature import extract_feature

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Re_id():
    def __init__(self):
        # self.data = Data()
        self.model = MGN()
        self.model.load_state_dict(torch.load("weights/re_id_model.pt", map_location='cpu'))

        self.test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model.eval()
        self.query_image = self.test_transform(default_loader('cache/query.jpg'))
        self.feature_a = extract_feature(self.model, tqdm([(torch.unsqueeze(self.query_image, 0), 1)]))

    def compare(self, query_image_path, input_image_path):
        # self.model.eval()

        # Extract feature
        print('extract features, this may take a few time')

        # query_image = self.test_transform(default_loader(query_image_path))
        input_image = self.test_transform(default_loader(input_image_path))

        # feature_a = extract_feature(self.model, tqdm([(torch.unsqueeze(query_image, 0), 1)]))
        feature_b = extract_feature(self.model, tqdm([(torch.unsqueeze(input_image, 0), 1)]))

        # sort images
        # feature_a = feature_a.view(-1, 1)
        feature_b = feature_b.view(-1, 1)

        # print(feature_b)

        # print(self.feature_a.size())

        score = torch.mm(self.feature_a, feature_b)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        return score





# class Main():
#     def __init__(self, model, loss, data):
#         self.train_loader = data.train_loader
#         self.test_loader = data.test_loader
#         self.query_loader = data.query_loader
#         self.testset = data.testset
#         self.queryset = data.queryset
#
#         # self.model = model.to('cuda')
#         self.model = model
#         self.loss = loss
#         self.optimizer = get_optimizer(model)
#         self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
#
#     def train(self):
#
#         self.scheduler.step()
#
#         self.model.train()
#         for batch, (inputs, labels) in enumerate(self.train_loader):
#             inputs = inputs.to('cuda')
#             labels = labels.to('cuda')
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.loss(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#
#     def evaluate(self):
#
#         self.model.eval()
#
#         print('extract features, this may take a few minutes')
#         qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
#         gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
#
#         def rank(dist):
#             r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
#                     separate_camera_set=False,
#                     single_gallery_shot=False,
#                     first_match_break=True)
#             m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
#
#             return r, m_ap
#
#         #########################   re rank##########################
#         q_g_dist = np.dot(qf, np.transpose(gf))
#         q_q_dist = np.dot(qf, np.transpose(qf))
#         g_g_dist = np.dot(gf, np.transpose(gf))
#         dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
#
#         r, m_ap = rank(dist)
#
#         print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
#               .format(m_ap, r[0], r[2], r[4], r[9]))
#
#         #########################no re rank##########################
#         dist = cdist(qf, gf)
#
#         r, m_ap = rank(dist)
#
#         print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
#               .format(m_ap, r[0], r[2], r[4], r[9]))
#
#     def vis(self):
#
#         self.model.eval()
#
#         gallery_path = data.testset.imgs
#         gallery_label = data.testset.ids
#
#         # Extract feature
#         print('extract features, this may take a few minutes')
#         query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
#         gallery_feature = extract_feature(model, tqdm(data.test_loader))
#
#         # sort images
#         query_feature = query_feature.view(-1, 1)
#         score = torch.mm(gallery_feature, query_feature)
#         score = score.squeeze(1).cpu()
#         score = score.numpy()
#
#         index = np.argsort(score)  # from small to large
#         index = index[::-1]  # from large to small
#
#         # # Remove junk images
#         # junk_index = np.argwhere(gallery_label == -1)
#         # mask = np.in1d(index, junk_index, invert=True)
#         # index = index[mask]
#
#         # Visualize the rank result
#         fig = plt.figure(figsize=(16, 4))
#
#         ax = plt.subplot(1, 11, 1)
#         ax.axis('off')
#         plt.imshow(plt.imread(opt.query_image))
#         ax.set_title('query')
#
#         print('Top 10 images are as follow:')
#
#         for i in range(10):
#             img_path = gallery_path[index[i]]
#             print(img_path)
#
#             ax = plt.subplot(1, 11, i + 2)
#             ax.axis('off')
#             plt.imshow(plt.imread(img_path))
#             ax.set_title(img_path.split('/')[-1][:9])
#
#         fig.savefig("show.png")
#         print('result saved to show.png')
#
#     def compare(self):
#
#         self.model.eval()
#
#         # Extract feature
#         print('extract features, this may take a few time')
#         feature_a = extract_feature(model, tqdm([(torch.unsqueeze(data.compare_img_a, 0), 1)]))
#         feature_b = extract_feature(model, tqdm([(torch.unsqueeze(data.compare_img_b, 0), 1)]))
#
#         # sort images
#         # feature_a = feature_a.view(-1, 1)
#         feature_b = feature_b.view(-1, 1)
#
#         print(feature_a)
#         print(feature_b)
#
#         print(feature_a.size())
#
#         score = torch.mm(feature_a, feature_b)
#         score = score.squeeze(1).cpu()
#         score = score.numpy()
#         print(score)
#
#
# if __name__ == '__main__':
#
#     data = Data()
#     model = MGN()
#     loss = Loss()
#     main = Main(model, loss, data)
#
#     print('compare')
#     model.load_state_dict(torch.load(opt.weight, map_location='cpu'))
#     main.compare()
