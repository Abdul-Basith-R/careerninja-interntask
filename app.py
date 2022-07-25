from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = Flask(__name__)
@app.route('/', methods =["GET", "POST"])
def home():
    if request.method == "POST":
       sentence1 = request.form.get("s1")
       sentence2 = request.form.get("s2")
       embedding_1 = model.encode(sentence1, convert_to_tensor=True)
       embedding_2 = model.encode(sentence2, convert_to_tensor=True)
       cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
       return str(cosine_similarity.item())

    return render_template("index.html")

@app.route('/request/<sentences>',methods=["GET","POST"])
def process_user(sentences):
    sentences = list(sentences.split(","))
#Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return str(cosine_similarity.item())

if __name__ == "__main__":
    app.run(debug=True)
