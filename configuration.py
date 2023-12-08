# Pre trained model path.
trained_model = r"pre_trained_model\BiLSTM-fake_news_detection.h5"

# Pickle file for tokenizer.
tokenizer_pickle_file = r"pickle\tokenizer.pkl"

# Newspaper image.
newspaper_img = r"/static/images/Newspaper_images.jpeg"

# Custom css file.
custom_css_file = r"/static/css/custom.css"

# Custoon js file.
js_file = r"/static/js/custom.js"

# Parameters for tokenizer
top_words = 6000  # Maximum vocabulary size
oov_tok = '<oov>'  # Out-of-vocabulary token

#  Parameters for padding.
max_news_length = 130
truncating = 'post'  # Truncate text at the end if it exceeds maxlen
padding = 'post'

# LIME
num_features = 50