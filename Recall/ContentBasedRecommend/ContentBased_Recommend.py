import numpy as np
import pandas as pd
import collections
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from tqdm import tqdm

class ContentBasedRecommend:
    def __init__(self):
        self.movie_profile = None
        self.inverted_table = {}
        self.user_profile = {}
        
    def load_data(self):
        """加载数据"""
        print("Loading tag data...")
        # 加载tag数据
        _tags = pd.read_csv("ml-latest-small/all-tags.csv", usecols=range(1, 3)).dropna()
        self.tags = _tags.groupby("movieId").agg(list)
        
        print("Loading movie data...")
        # 加载电影数据
        movies = pd.read_csv("ml-latest-small/movies.csv")
        movies['genres'] = movies['genres'].map(lambda x: x.split('|'))
        self.movies = movies.set_index('movieId')
        
        print("Loading user watch records...")
        # 加载用户观看记录
        self.watch_record = pd.read_csv("ml-latest-small/ratings.csv", 
                                      usecols=range(2), 
                                      dtype={"userId":np.int32, "movieId": np.int32})
        self.watch_record = self.watch_record.groupby("userId").agg(list)

    def create_movie_profile(self):
        """构建电影画像"""
        print("Merging tags and genres...")
        # 合并tags和genres
        movie_profile = pd.concat([self.tags, self.movies], axis=1)
        movie_profile['tag'] = movie_profile['tag'].fillna('').apply(lambda x: [] if x == '' else x)
        
        print("Calculating TF-IDF...")
        # 计算TF-IDF
        vectorizer = TfidfVectorizer(lowercase=False)
        
        # 处理NaN值并连接tag和genres
        movie_profile['combined'] = movie_profile.apply(
            lambda x: ' '.join(x['tag'] if isinstance(x['tag'], list) else [] + 
                             x['genres'] if isinstance(x['genres'], list) else []), axis=1
        )
        
        tfidf_matrix = vectorizer.fit_transform(movie_profile['combined'])
        
        # 获取关键词和权重
        words = vectorizer.get_feature_names_out()
        weights = tfidf_matrix.toarray()
        
        print("Building movie profiles...")
        # 构建电影画像
        profiles = []
        for idx, weight in tqdm(enumerate(weights), total=len(weights), desc="Processing movies"):
            # 获取权重前30的关键词
            sorted_words = sorted(zip(words, weight), key=lambda x: x[1], reverse=True)[:30]
            profile = [w for w, _ in sorted_words]
            weight_dict = {w: float(v) for w, v in sorted_words}
            profiles.append({
                'profile': profile,
                'weights': weight_dict
            })
            
        self.movie_profile = pd.DataFrame(profiles, 
                                        index=movie_profile.index,
                                        columns=['profile', 'weights'])
        self.movie_profile = pd.concat([self.movies['title'], self.movie_profile], axis=1)

    def create_inverted_table(self):
        """建立倒排索引"""
        for mid, row in tqdm(self.movie_profile.iterrows(), total=len(self.movie_profile), desc="Building inverted index"):
            weights = row['weights']
            for tag, weight in weights.items():
                self.inverted_table.setdefault(tag, []).append((mid, weight))

    def create_user_profile(self, top_n=10):
        """构建用户画像"""
        for uid, movies in tqdm(self.watch_record.iterrows(), total=len(self.watch_record), desc="Creating user profiles"):
            watched_movies = movies['movieId']
            interest_words = []
            
            # 收集用户看过的电影的所有关键词
            for mid in watched_movies:
                if mid in self.movie_profile.index:
                    interest_words.extend(self.movie_profile.loc[mid, 'profile'])
            
            # 统计关键词频率
            word_count = collections.Counter(interest_words)
            
            # 选择top_n个关键词作为用户兴趣
            self.user_profile[uid] = word_count.most_common(top_n)

    def recommend(self, user_id, top_k=10):
        """为用户生成推荐"""
        if user_id not in self.user_profile:
            print(f"User {user_id} not found in user profiles")
            return []
            
        result_table = {}
        
        # 获取用户兴趣词对应的电影
        for interest_word, interest_weight in self.user_profile[user_id]:
            related_movies = self.inverted_table.get(interest_word, [])
            for mid, relate_weight in related_movies:
                result_table.setdefault(mid, []).append(interest_weight * relate_weight)
        
        # 计算最终得分并排序
        movie_scores = [(mid, sum(weights)) for mid, weights in result_table.items()]
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k推荐结果，添加检查确保电影ID存在
        recommendations = []
        for mid, score in movie_scores[:top_k]:
            if mid in self.movies.index:
                recommendations.append((mid, score, self.movies.loc[mid, 'title']))
        
        if not recommendations:
            print("No recommendations found for this user")
        return recommendations

    def run(self):
        """运行推荐流程"""
        print("Loading data...")
        self.load_data()
        
        print("Creating movie profiles...")
        self.create_movie_profile()
        
        print("Building inverted index...")
        self.create_inverted_table()
        
        print("Creating user profiles...")
        self.create_user_profile()
        
        return self

if __name__ == "__main__":
    # 创建推荐器实例
    recommender = ContentBasedRecommend()
    
    # 运行推荐流程
    recommender.run()
    
    # 为用户1生成推荐
    recommendations = recommender.recommend(1)
    
    if recommendations:  # 只有在有推荐结果时才打印
        print("\nRecommendations for user 1:")
        for movie_id, score, title in recommendations:
            print(f"Movie: {title}, Score: {score:.4f}")