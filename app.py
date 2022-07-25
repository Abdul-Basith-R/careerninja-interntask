from crypt import methods
from flask import Flask, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/request/<sentences>',methods=["POST"])
def process_user(sentences):
    sentences = list(sentences.split(","))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return str(cosine_similarity.item())

if __name__ == "__main__":
    app.run(debug=True)
