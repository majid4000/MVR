# pathes
# raw data pathes
ROOT_DATA = "/Users/zhouyf/Documents/data/majid/drive/MyDrive/project2/data"
result_path = f"{ROOT_DATA}/result"
raw_data_path = f'{ROOT_DATA}/raw_data'
tweets_x = f"{ROOT_DATA}/raw_data/Tweet.txt"
tweets_y = f"{ROOT_DATA}/raw_data/Tweet_LABEL.txt"
googel_news_x = f"{ROOT_DATA}/raw_data/GoogleNews.txt"
googel_news_y = f"{ROOT_DATA}/raw_data/GoogleNews_LABEL.txt"
stack_overflow_x = f"{ROOT_DATA}/raw_data/StackOverflow.txt"
stack_overflow_y = f"{ROOT_DATA}/raw_data/StackOverflow_label.txt"

# organized data
organized_data_path = f"{ROOT_DATA}/organized_data"
tweets_og = f'{organized_data_path}/TweetData.json'
googel_news_og = f'{organized_data_path}/GoogleNewsData.json'
stack_overflow_og = f'{organized_data_path}/StackOverflowData.json'

# organized cleaned data pathes
data_path = f'{ROOT_DATA}/cleaned_data'
tweets = f'{data_path}/TweetData.json'
googel_news = f'{data_path}/GoogleNewsData.json'
stack_overflow = f'{data_path}/StackOverflowData.json'

# assets
goole_label_map = f'{ROOT_DATA}/assets/google_label_map.json'
google_inv_label_map = f'{ROOT_DATA}/assets/google_inv_label_map.json'


stackOverflow_label_map = f'{ROOT_DATA}/assets/StackOverflow_label_map.json'
stackOverflow_inv_label_map = f'{ROOT_DATA}/assets/StackOverflow_inv_label_map.json'

tweet_label_map = f'{ROOT_DATA}/assets/tweet_label_map.json'
tweet_inv_label_map = f'{ROOT_DATA}/assets/tweet_inv_label_map.json'

embed_path = f"{ROOT_DATA}/embed"

# cnn model
cnn_model_dir = f'{ROOT_DATA}/model/cnn'

# doc to vec model
# https://cloudstor.aarnet.edu.au/plus/s/hpfhUC72NnKDxXw
