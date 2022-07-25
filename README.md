# CareerNinja-interntask

### Assignment:<br>
Build a tool that checks the similarity of two sentences with the similarity score measure using any word embedding (pre-trained models are also fine). And the model should be deploy-able and accessible via API (flask or fastAPI).
Input: Two sentences
Output: Similarity score (0-1)

Code Implementation:
I have created a flask application that can accept api request that processes and returns the similarity score as string. We can also manually enter the two sentences through the application and see the similarity score.

For Sentence similarity checking I have used hugging face sentence transformers framework which accepts the value as strings and converts it to tensors and returns a similarity score. I have used MiniLM-L6-v2 model from hugging face.
