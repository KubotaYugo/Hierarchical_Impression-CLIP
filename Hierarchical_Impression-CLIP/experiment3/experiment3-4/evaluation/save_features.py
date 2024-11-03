'''
学習済みモデルで抽出したtrain/val/testの画像特徴/印象特徴(複数タグ), 印象特徴(タグ単体)を保存
'''
from transformers import CLIPTokenizer, CLIPModel
import torch
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from models import FontAutoencoder
from models import MLP
from lib import utils
from lib import eval_utils


# define constant
EXP = utils.EXP
IMG_CLUSTER_PATH = utils.IMG_CLUSTER_PATH
TAG_CLUSTER_PATH = utils.TAG_CLUSTER_PATH
BATCH_SIZE = utils.BATCH_SIZE
BASE_DIR = utils.BASE_DIR
MODEL_PATH = utils.MODEL_PATH

# モデルの準備
device = torch.device('cuda:0')
font_autoencoder = FontAutoencoder.Autoencoder(FontAutoencoder.ResidualBlock, [2, 2, 2, 2]).to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
emb_i = MLP.ReLU().to(device)
emb_t = MLP.ReLU().to(device)
font_autoencoder.eval()
clip_model.eval()
emb_i.eval()
emb_t.eval()


# パラメータの読み込み
params = torch.load(MODEL_PATH)
font_autoencoder.load_state_dict(params['font_autoencoder'])
clip_model.load_state_dict(params['clip_model'])
emb_i.load_state_dict(params['emb_i'])
emb_t.load_state_dict(params['emb_t'])

for DATASET in ['train', 'val', 'test']:
    # ディレクトリの作成
    SAVE_DIR = f'{BASE_DIR}/features/{DATASET}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # dataloderの準備
    img_paths, tag_paths = utils.load_dataset_paths(DATASET)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = eval_utils.DMH_D_Eval(img_paths, tag_paths, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    # 単体タグのdataloder
    tag_list = list(eval_utils.get_tag_list().values())
    tagset = eval_utils.DMH_D_ForTag(tag_list, tokenizer)
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