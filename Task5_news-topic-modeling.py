if __name__ == "__main__":
    import pandas as pd
    import nltk
    import spacy
    from nltk.corpus import stopwords
    from gensim import corpora
    from gensim.models import LdaModel, CoherenceModel
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    import pyLDAvis.gensim_models
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    # -----------------------------
    # Download NLTK resources
    # -----------------------------
    nltk.download('stopwords')

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv(r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task5\Data_Set\BBCNews.csv")
    print("Dataset Loaded. Sample:")
    print(df.head())

    # -----------------------------
    # Initialize spaCy
    # -----------------------------
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # -----------------------------
    # Preprocessing Function
    # -----------------------------
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
        return tokens

    # Apply preprocessing
    df['tokens'] = df['descr'].apply(preprocess)
    print("\nSample tokens:")
    print(df[['descr', 'tokens']].head())

    # -----------------------------
    # Create Dictionary and Corpus for LDA
    # -----------------------------
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]

    # -----------------------------
    # LDA Topic Modeling
    # -----------------------------
    num_topics = 5
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                         random_state=42, passes=10, alpha='auto', per_word_topics=True)

    print("\nLDA Topics:")
    for i, topic in lda_model.show_topics(formatted=True, num_words=10):
        print(f"Topic {i}: {topic}")

    # -----------------------------
    # LDA Visualization
    # -----------------------------
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, 'lda_visualization.html')
    print("\nLDA visualization saved to 'lda_visualization.html'.")

    # -----------------------------
    # WordClouds for LDA Topics
    # -----------------------------
    for i in range(num_topics):
        topic_words = dict(lda_model.show_topic(i, topn=50))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"LDA Topic {i}")
        plt.tight_layout()
        plt.savefig(f"LDA_Topic_{i}.png")
        plt.close()
    print("✅ LDA WordClouds saved automatically.")

    # -----------------------------
    # NMF Topic Modeling using TF-IDF
    # -----------------------------
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(df['descr'])

    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(tfidf)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("\nNMF Topics:")

    for i, topic in enumerate(nmf_model.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]
        print(f"Topic {i}: {', '.join(top_words)}")

        # WordCloud for NMF (saved automatically)
        word_freq = {feature_names[j]: topic[j] for j in range(len(feature_names))}
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"NMF Topic {i}")
        plt.tight_layout()
        plt.savefig(f"NMF_Topic_{i}.png")
        plt.close()
    print("✅ NMF WordClouds saved automatically.")

    # -----------------------------
    # LDA vs NMF Performance
    # -----------------------------
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
    lda_coherence = coherence_model_lda.get_coherence()
    print(f"\nLDA Coherence Score: {lda_coherence:.4f}")

    nmf_reconstruction_error = nmf_model.reconstruction_err_
    print(f"NMF Reconstruction Error: {nmf_reconstruction_error:.4f}")

    print("\n✅ Task 5 completed successfully.")
