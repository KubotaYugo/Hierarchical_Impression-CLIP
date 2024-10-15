'''
学習済みモデルで抽出したtrain/val/testの画像特徴/印象特徴(複数タグ), 印象特徴(タグ単体)を保存
'''

from transformers import CLIPTokenizer, CLIPModel
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import os

import FontAutoencoder
import MLP
import utils
import eval_utils


# define constant
EXP = utils.EXP
BATCH_SIZE = utils.BATCH_SIZE
MODEL_PATH = f"{EXP}/model/best.pth.tar"

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
font_encoder = font_autoencoder.encoder    
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)


# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_encoder.load_state_dict(params['font_encoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()


for DATASET in ['train', 'val', 'test']:
    # ディレクトリの作成
    SAVE_DIR = f'{EXP}/features/{DATASET}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # dataloderの準備
    img_paths, tag_paths = utils.LoadDatasetPaths(DATASET)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = utils.CustomDataset(img_paths, tag_paths, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    # 単体タグのdataloder
    tag_list = list(eval_utils.get_tag_list().values())
    tagset = eval_utils.CustomDatasetForTag(tag_list, tokenizer)
    tagloader = torch.utils.data.DataLoader(tagset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 画像特徴，印象特徴(複数タグ)の保存
    img_features, tag_features, embedded_img_features, embedded_tag_features = eval_utils.extract_features(font_autoencoder, clip_model, emb_i, emb_t, dataloader)
    torch.save(img_features, f'{SAVE_DIR}/img_features.pth')
    torch.save(tag_features, f'{SAVE_DIR}/tag_features.pth')
    torch.save(embedded_img_features, f'{SAVE_DIR}/embedded_img_features.pth')
    torch.save(embedded_tag_features, f'{SAVE_DIR}/embedded_tag_features.pth')

    # 印象特徴(タグ単体)の保存
    single_tag_features, embedded_single_tag_features = eval_utils.extract_text_features(tagloader, clip_model, emb_t)
    torch.save(single_tag_features, f'{SAVE_DIR}/single_tag_features.pth')
    torch.save(embedded_single_tag_features, f'{SAVE_DIR}/embedded_single_tag_features.pth')