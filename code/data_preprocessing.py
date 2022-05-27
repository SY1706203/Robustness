import clip
import json
import gzip
import torch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import preprocessing

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import data
data_item = []
with gzip.open('/home/stud/zongyue/workarea/Robustness-1/data/meta_Gift_Cards.json.gz') as f:
    for l in f:
        data_item.append(json.loads(l.strip()))

# total length of list, this number equals total number of products
print(len(data_item))

df_item = pd.DataFrame.from_dict(data_item)

data_user = []
with gzip.open('C:/Users/Zongyue Li/Downloads/Gift_Cards.json.gz') as f:
    for l in f:
        data_user.append(json.loads(l.strip()))

df_user = pd.DataFrame.from_dict(data_user)

# Adj construction

# k = 1
# for asin in df_item['asin']:
#     df_user.loc[df_user['asin'] == asin, 'idx'] = k
#     df_item.loc[df_item['asin'] == asin, 'idx'] = k
#     k = k + 1
#     # print(df_user['idx'])
# df_item.index = df_item.index.astype(int)
# df_user.index = df_user.index.astype(int)

# UserItemNet = csr_matrix((np.ones(len(df_user)), (df_user.index, df_user['idx'])),
#                                      shape=(len(df_user["reviewerID"].unique()), len(df_item)))


# items pre-processing

item_features = None
if 'category' in df_item:
    mle = preprocessing.MultiLabelBinarizer()
    cate_unique = df_item['category'].explode().unique()
    item_features = mle.fit_transform(df_item['category'])

# user pre-processing

df_user_review = df_user[['asin', 'reviewerID', 'reviewText']].dropna()
# df_user_mask = df_user.asin.isin(asin)

# df_user_filtered = df_user[df_user.asin.isin(asin)]
# df_user_review = df_user_filtered[['asin', 'reviewText']]

print('finished')

# model, preprocess = clip.load("ViT-B/32", device=device)

review_list = df_user_review['reviewText'].str.replace(',', ' ')
review_list = review_list.str.replace('.', ' ')
review_list = review_list.str.replace('!', ' ')
review_list = review_list.str.replace('?', ' ')
review_list = review_list.str.replace('\n', ' ')
review_list = review_list.str.split(' ')
df_user_review['length'] = review_list.apply(len)

user_features = []
# emb_len = model.context_length

# df_user_review = df_user_review[df_user_review.length <= emb_len]

user_idx = 0
for user in df_user_review['reviewerID'].unique():
    review_embedding = []
    df_user_review.loc[df_user_review['reviewerID'] == user, 'user_idx'] = user_idx
    # for text in df_user_review[df_user_review['reviewerID'] == user]['reviewText'].tolist():
    #     review_embedding.append(model.encode_text(
    #         clip.tokenize(texts=text, context_length=model.context_length, truncate=True).to(
    #             device)).cpu().detach().numpy().squeeze(0))
    # each_review = np.array(review_embedding)
    # each_review = np.mean(each_review, 0, keepdims=True)
#
    # user_features.append(each_review.squeeze(0))
    user_idx = user_idx + 1

item_idx = 0
for asin in df_item['asin']:
    df_user_review.loc[df_user_review['asin'] == asin, 'item_idx'] = item_idx
    item_idx = item_idx + 1

# NaN appears in df_user_review['item_idx'].
# Reason: some users bought items out of df_item. (df_user_review['asin'] == asin) leads to all False.
df_user_review = df_user_review.dropna()

# torch.save(torch.tensor(item_features), '/home/stud/zongyue/workarea/Robustness-1/data/item_feature_tensor.pt')
# torch.save(torch.tensor(user_features), '/home/stud/zongyue/workarea/Robustness-1/data/user_feature_tensor.pt')

UserItemNet = csr_matrix((np.ones(len(df_user_review)), (df_user_review['user_idx'], df_user_review['item_idx'])),
                         shape=(len(df_user_review["reviewerID"].unique()), len(df_item)))
