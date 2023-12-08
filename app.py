from flask import Flask,render_template,request
from waitress import server, serve

import functions
import configuration as conf

# Read configuration variables from configuration file.
newspaper_img = conf.newspaper_img
css_file = conf.custom_css_file
js_file = conf.js_file

app = Flask(__name__)

# Render home page.
@app.route('/')
def index():
    return render_template('index.html',
                            newspaper_img=newspaper_img,
                            css_file=css_file)

# Make prediction.
@app.route('/predict',methods=["GET","POST"])
def predict():
    if request.method == 'POST':

        # Read input fom the webpage.
        raw_news_text = request.form['rawtext']
        news_text = [raw_news_text.strip()]
        news_text = [i for i in news_text if i]

        # Keep only first 150 words.
        news_text_150_words = ' '.join(news_text[0].split()[:150])

        # Clean the text.
        cleaned_news_text, word_mapping = functions.process_text(news_text_150_words)

        # Tokenize and pad text.
        padded_news_text = functions.tokenize_pad_text(cleaned_news_text)

        # Make prediction on cleaned data.
        prediction_result = functions.make_prediction(padded_news_text)

        # LIME Explanation.
        word_weights = functions.lime_explanation(cleaned_news_text, prediction_result)

        # Round all the weights to 5 digit.
        word_weights_rounded = {word: round(weight, 5) for word, weight in word_weights.items()}

        # Highlight text.
        highlighted_text = functions.text_highlighter(news_text_150_words, word_weights_rounded, word_mapping)

        # Define words and weights based on your data
        words = list(word_weights_rounded.keys())
        weights = list(word_weights_rounded.values())

    newspaper_img = conf.newspaper_img

    # Find out sentiment and tone of the news.
    news_sentiment, news_tone, err_msg = functions.sentiment_tone_analysis(news_text_150_words)
    ai_detection = functions.ai_detection(news_text_150_words)
    news_type = functions.NewsType_detection(news_text_150_words)
    news_intent = functions.NewsIntent_detection(news_text_150_words)
    print("news Intent is " +str(news_intent[0]))
    # print("Ai detection is" +str(ai_detection[0]))
    # Relevant news fetch.
    clean_text = functions.clean_text(news_text_150_words)
    first_20_words = ' '.join(clean_text.split()[:20])
    relevant_news_dict = functions.fetch_rss_feed(first_20_words)

    # Render result page.
    return render_template('result.html',
                            words=words,
                            weights=weights,
                            rawtext=highlighted_text,
                            prediction_result=prediction_result,
                            newspaper_img=newspaper_img,
                            css_file=css_file,
                            js_file=js_file,
                            word_weights=word_weights_rounded,
                            news_sentiment=news_sentiment,
                            news_tone=news_tone,
                            relevant_news_dict=relevant_news_dict,
                            ai_detection = ai_detection[0],
                            news_type = news_type[0],
                            news_intent = news_intent[0])

if __name__ == "__main__":
    serve(app)
