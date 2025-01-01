from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
clean_data=pd.read_csv('cleaned_movie_data.csv')
cv=CountVectorizer(max_features=500,stop_words='english')
vectors=cv.fit_transform(clean_data['tags']).toarray()
similarity=cosine_similarity(vectors)

app=Flask(__name__)

def recomended_movie(movie,n_movies=5):
    if movie not in clean_data['title'].values:
        return []
    
    movie_index=clean_data[clean_data['title']==movie].index[0]
    distances=similarity[movie_index]
    enumlist=list(enumerate(distances))
    sorted_list=sorted(enumlist,reverse=True,key=lambda x:x[1])[1:n_movies]
    recommended_indeces=[i[0] for i in sorted_list]
    return clean_data.iloc[recommended_indeces]['title'].values
    

@app.route('/',methods=['GET','POST'])
def index():
    recomendations=[]
    if request.method=='POST':
        user_input=request.form.get('user_input')
        recomendations=recomended_movie(user_input)
    return render_template('index.html',Recommendations=recomendations)

#recommendation system

if __name__=='__main__':
    app.run(debug=True)
