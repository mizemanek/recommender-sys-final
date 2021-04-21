'''
Create content-based recommenders: Feature Encoding, TF-IDF/CosineSim
       using item/genre feature data
       

Programmer name: << Your name here!!>>

Collaborator/Author: Carlos Seminario

sources: 
https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XoT9p257k1L

reference:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html


'''

import numpy as np
import pandas as pd
import math
import os
import copy
import pickle
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIG_THRESHOLD = 0 # accept all positive similarities > 0 for TF-IDF/ConsineSim Recommender
                  # others: TBD ...
    
def from_file_to_2D(path, genrefile, itemfile):
    ''' Load feature matrix from specified file 
        Parameters:
        -- path: directory path to datafile and itemfile
        -- genrefile: delimited file that maps genre to genre index
        -- itemfile: delimited file that maps itemid to item name and genre
        
        Returns:
        -- movies: a dictionary containing movie titles (value) for a given movieID (key)
        -- genres: dictionary, key is genre, value is index into row of features array
        -- features: a 2D list of features by item, values are 1 and 0;
                     rows map to items and columns map to genre
                     returns as np.array()
    
    '''
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    ##
    # Get movie genre from the genre file, place into genre dictionary indexed by genre index
    genres={} # key is genre index, value is the genre string
    try:
        with open (path + '/' + genrefile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (val,key)=line.split('|')
                key = key.replace('\n', '')
                genres[int(key)]=val
    
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(genres), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(genres), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(genres))
        return {}
    ##
    ## Your code here!!
    ##
    
    # Load data into a nested 2D list
    features = []
    start_feature_index = 5
    try: 
        for line in open(path+'/'+ itemfile, encoding='iso8859'):
            #print(line, line.split('|')) #debug
            fields = line.split('|')[start_feature_index:]
            row = []
            for feature in fields:
                row.append(int(feature))
            features.append(row)
        features = np.array(features)
    except Exception as ex:
        print (ex)
        print ('Proceeding with len(features)', len(features))
        #return {}
    
    #return features matrix
    return movies, genres, features    

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings (value) for each user (key)
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except ValueError as ex:
        print ('ValueError', ex)
        print (len(movies), line, id, title)
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}

    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def prefs_to_2D_list(prefs):
    '''
    Convert prefs dictionary into 2D list used as input for the MF class
    
    Parameters: 
        prefs: user-item matrix as a dicitonary (dictionary)
        
    Returns: 
        ui_matrix: (list) contains user-item matrix as a 2D list
        
    '''
    ui_matrix = []
    
    user_keys_list = list(prefs.keys())
    num_users = len(user_keys_list)
    #print (len(user_keys_list), user_keys_list[:10]) # debug
    
    itemPrefs = transformPrefs(prefs) # traspose the prefs u-i matrix
    item_keys_list = list(itemPrefs.keys())
    num_items = len(item_keys_list)
    #print (len(item_keys_list), item_keys_list[:10]) # debug
    
    sorted_list = True # <== set manually to test how this affects results
    
    if sorted_list == True:
        user_keys_list.sort()
        item_keys_list.sort()
        print ('\nsorted_list =', sorted_list)
        
    # initialize a 2D matrix as a list of zeroes with 
    #     num users (height) and num items (width)
    
    for i in range(num_users):
        row = []
        for j in range(num_items):
            row.append(0.0)
        ui_matrix.append(row)
          
    # populate 2D list from prefs
    # Load data into a nested list

    for user in prefs:
        for item in prefs[user]:
            user_idx = user_keys_list.index(user)
            movieid_idx = item_keys_list.index(item) 
            
            try: 
                # make it a nested list
                ui_matrix[user_idx][movieid_idx] = prefs [user][item] 
            except Exception as ex:
                print (ex)
                print (user_idx, movieid_idx)   
                
    # return 2D user-item matrix
    return ui_matrix

def to_array(prefs):
    ''' convert prefs dictionary into 2D list '''
    R = prefs_to_2D_list(prefs)
    R = np.array(R)
    print ('to_array -- height: %d, width: %d' % (len(R), len(R[0]) ) )
    return R

def to_string(features):
    ''' convert features np.array into list of feature strings '''
    
    feature_str = []
    for i in range(len(features)):
        row = ''
        for j in range(len (features[0])):
            row += (str(features[i][j]))
        feature_str.append(row)
    print ('to_string -- height: %d, width: %d' % (len(feature_str), len(feature_str[0]) ) )
    return feature_str

def to_docs(features_str, genres):
    ''' convert feature strings to a list of doc strings for TFIDF '''
    
    feature_docs = []
    for doc_str in features_str:
        row = ''
        for i in range(len(doc_str)):
            if doc_str[i] == '1':
                row += (genres[i] + ' ') # map the indices to the actual genre string
        feature_docs.append(row.strip()) # and remove that pesky space at the end
        
    print ('to_docs -- height: %d, width: varies' % (len(feature_docs) ) )
    return feature_docs

def cosine_sim(docs):
    ''' Perofmrs cosine sim calcs on features list, aka docs in TF-IDF world
    
        Parameters:
        -- docs: list of item features
     
        Returns:   
        -- list containing cosim_matrix: item_feature-item_feature cosine similarity matrix 
    
    
    '''
    
    print()
    print('## Cosine Similarity calc ##')
    print()
    print('Documents:', docs[:10])
    
    print()
    print ('## Count and Transform ##')
    print()
    
    # choose one of these invocations
    tfidf_vectorizer = TfidfVectorizer() # orig
  
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    #print (tfidf_matrix.shape, type(tfidf_matrix)) # debug

    
    print()
    print('Document similarity matrix:')
    cosim_matrix = cosine_similarity(tfidf_matrix[0:], tfidf_matrix)
    print (type(cosim_matrix), len(cosim_matrix))
    print()
    print(cosim_matrix[0:6])
    print()
    
    '''
    print('Examples of similarity angles')
    if tfidf_matrix.shape[0] > 2:
        for i in range(6):
            cos_sim = cosim_matrix[1][i] #(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))[0][i] 
            if cos_sim > 1: cos_sim = 1 # math precision creating problems!
            angle_in_radians = math.acos(cos_sim)
            print('Cosine sim: %.3f and angle between documents 2 and %d: ' 
                  % (cos_sim, i+1), end=' ')
            print ('%.3f degrees, %.3f radians' 
                   % (math.degrees(angle_in_radians), angle_in_radians))
    '''
    
    return cosim_matrix

def movie_to_ID(movies):
    ''' converts movies mapping from "id to title" to "title to id" '''
    inv_movies = {v: k for k, v in movies.items()}
    return inv_movies
    pass

def get_TFIDF_recommendations(prefs,cosim_matrix,user,movie_title_to_id,sim_threshold):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- cosim_matrix: list containing item_feature-item_feature cosine similarity matrix 
        -- user: string containing name of user requesting recommendation        
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    n = input("Enter a value for n, the number of top rankings to select: ")
    """
    sim_threshold = input("Choose a Similarity Threshold (0,.340,.530,.903,1): ")
    if int(sim_threshold) > 1 or int(sim_threshold) < 0:
        sim_threshold = input("Similarity Threshold must be between 0 and 1. Please try again: ")
    """
    movie_to_movie_id = {}
    for name_id in prefs:
        for movie in prefs[name_id]:
            movie_to_movie_id[(int(movie_title_to_id.get(movie)))] = movie
    
    for name_id in prefs:
        total_ratings = []
        movie_id_to_rating = {}
        user_rated = []
        unknown_ratings = []
        
        for movie in prefs[user]:
            user_rated.append(int(movie_title_to_id.get(movie)))
            movie_id_to_rating[int(movie_title_to_id.get(movie))] = prefs[user].get(movie)
            
        for num in range(1,len(cosim_matrix)+1):
            if num not in user_rated:
                unknown_ratings.append(num)
        
        for m_id in unknown_ratings:
            if type(movie_to_movie_id.get(m_id)) == str:
                movie_name = movie_to_movie_id.get(m_id)
            numerator = []
            denominator = []
            
            for sim in range(len(cosim_matrix[int(m_id)-1])):
                if cosim_matrix[(int(m_id))-1][sim] >= float(sim_threshold):
                    if (int(sim)+1) in movie_id_to_rating:
                        numerator.append(cosim_matrix[int(m_id)-1][sim] * movie_id_to_rating.get(float(sim+1)))
                        denominator.append(cosim_matrix[int(m_id)-1][sim])
            if sum(denominator) != 0:
                total_ratings.append([(sum(numerator)/sum(denominator)),movie_name])
        
        total_ratings = (sorted(total_ratings, key = lambda x: x[0], reverse = True))
        return(total_ratings[0:int(n)])

def get_FE_recommendations(prefs, features, movie_title_to_id, user):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- features: an np.array whose height is based on number of items
                     and width equals the number of unique features (e.g., genre)
        -- movie_title_to_id: dictionary that maps movie title to movieid
        -- user: string containing name of user requesting recommendation        
        
        Returns:
        -- ranknigs: A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    num_feat = 0
    path = ''
    item_file = ''
    if len(prefs) > 10:
        num_feat = 19
        path = 'data/ml-100k/'
        item_file = 'u.item'
    else:
        num_feat = 12
        path = 'data/'
        item_file = 'critics_movies.item'

    #Create the MOVIE-FEATURE matrix
    counter = 0
    path = 'data/ml-100k/'
    with open (path + '/' + item_file, encoding='iso8859') as myfile: 
        # this encoding is required for some datasets: encoding='iso8859'
        for line in myfile:
            (line_data)=line.split('|')
            line_data = line_data[-num_feat:]
            features[counter] = line_data
            counter += 1
    
    #Create a users feature profile
    feature_preference = {}
    for item in prefs[user]:
        feature_preference[item] = features[int(movie_title_to_id[item])-1]

    
    #Multiply features by their associated ratings
    for item in prefs[user].items(): #item[0] is movie str, item[1] is rating
        if(item[0] != None):
            for i in range(len(feature_preference[item[0]])):
                feature_preference[item[0]][i] *= item[1]
                
    #Create totals vector
    totals = [0] * num_feat 
    for arr in feature_preference.values():
        totals = np.add(totals, arr)


    #Create feature frequency vector
    rated_feature_freq = [0] * num_feat 
    for arr in feature_preference.values():
        for j in range(len(arr)):
            if(arr[j] != 0):
                rated_feature_freq[j] += 1
    
    #Calc normalized vector
    overall_sum = np.sum(totals)
    normalized_vector = totals/overall_sum
    
    
    #Create a list of rated movie ids
    rated_movies = prefs[user].keys()
    rated_ids = []
    for item in rated_movies:
        rated_ids.append(movie_title_to_id[item])
        
    #Create a prediction for unrated items
    curr_id = 1
    pred = []
    for arr in features:
        if curr_id not in list(rated_ids) and (True in np.logical_and(features[curr_id-1], totals)):
            array_to_sum = np.multiply(arr, normalized_vector) #hadamard
            
            summed = np.sum(array_to_sum)
            if(summed != 0):
                normalized_weight = array_to_sum / summed
            avg_rating_arr = totals / rated_feature_freq
            components =  np.multiply(avg_rating_arr, normalized_weight)


            pred.append(( np.nansum(components), curr_id))
            
        curr_id += 1
    
    pred.sort(reverse=True)
    
    
    return pred
def sim_distance(prefs, person1, person2, sim_weighting=0):
    '''
        Calculate Euclidean distance similarity
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- sim_weighting: similarity significance weighting factor (0, 25, 50), 
                            default is 0 [None]
        Returns:
        -- Euclidean distance similarity as a float
    '''

    # Get the list of shared_items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])

    distance_sim = 1/(1+np.sqrt(sum_of_squares))

    # apply significance weighting, if any

    if sim_weighting != 0:
        if len(si) < sim_weighting:
            distance_sim *= (len(si) / sim_weighting)

    return distance_sim

def sim_pearson(prefs, p1, p2, sim_weighting=0):
    '''
        Calculate Pearson Correlation similarity
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        -- sim_weighting: similarity significance weighting factor (0, 25, 50), 
                            default is 0 [None]
        Returns:
        -- Pearson Correlation similarity as a float
    '''

    # Get the list of shared_items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

    # if they have no ratings in common, return 0
    if len(si) == 0:
        return 0

    # sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
        # for item in prefs[person1] if item in prefs[person2]])

    # calc avg rating for p1 and p2, using only shared ratings
    x_avg = 0
    y_avg = 0

    for item in si:
        x_avg += prefs[p1][item]
        y_avg += prefs[p2][item]

    x_avg /= len(si)
    y_avg /= len(si)

    # calc numerator of Pearson correlation formula
    numerator = sum([(prefs[p1][item] - x_avg) * (prefs[p2][item] - y_avg)
                     for item in si])

    # calc denominator of Pearson correlation formula
    denominator = math.sqrt(sum([(prefs[p1][item] - x_avg)**2 for item in si])) * \
        math.sqrt(sum([(prefs[p2][item] - y_avg)**2 for item in si]))

    # catch divide-by-0 errors
    if denominator != 0:
        sim_pearson = numerator / denominator

        # apply significance weighting, if any
        if sim_weighting != 0:
            sim_pearson *= (len(si) / sim_weighting)

        return sim_pearson
    else:
        return 0
    
def topMatches(prefs, person, similarity=sim_pearson, n=5, sim_weighting=0, sim_threshold=0):
    '''
        Returns the best matches for person from the prefs dictionary
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        -- sim_weighting: similarity significance weighting factor (0, 25, 50), 
                            default is 0 [None]
        Returns:
        -- A list of similar matches with 0 or more tuples,
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
    '''
    scores = []
    for other in prefs:
        score = similarity(prefs, person, other, sim_weighting)
        if other != person and score > sim_threshold:
            scores.append((score, other))

    # scores = [(similarity(prefs, person, other, sim_weighting), other)
            #   for other in prefs if other != person]

    scores.sort()
    scores.reverse()
    return scores[0:n]
    
def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary)
        Parameters:
        -- prefs: dictionary containing user-item matrix
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix,
           this function returns an I-U matrix
    '''

    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            # Flip item and person
            result[item][person] = prefs[person][item]
    return result

def calculateSimilarItems(prefs, n=100, similarity=sim_pearson, sim_weighting=0, sim_threshold=0):
    '''
        Creates a dictionary of items showing which other items they are most
        similar to.
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        -- sim_weighting: similarity significance weighting factor (0, 25, 50), 
                            default is 0 [None]
        Returns:
        -- A dictionary with a similarity matrix
    '''

    result = {}
    c = 0

    # Invert the preference matrix to be item-centric
    itemPrefs = transformPrefs(prefs)

    for item in itemPrefs:
      # Status updates for larger datasets
        c += 1
        if c % 100 == 0:
            percent_complete = (100*c)/len(itemPrefs)
            print(str(percent_complete)+"% complete")

        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, similarity, n,
                            sim_weighting, sim_threshold)
        result[item] = scores
    return result

def update_Cosim_Matrix(movie_title_to_id,itemsim,cosim_matrix,weight_factor):
    updated_matrix = np.copy(cosim_matrix)
    movie_id = []
    m_id = []
    for movie in itemsim:
        movie_id.append(movie_title_to_id.get(movie))
    
    for movie in itemsim:
        name_sim = {}
        num = []
        for j in (itemsim[movie]):
            sim, name = j
            name_sim[movie_title_to_id.get(name)] = sim
        for i in movie_id:
            if i not in name_sim:
                name_sim[i] = 0
        
        for key in range(1,len(itemsim)+1):
            num.append(name_sim.get(str(key)))
        m_id.append(num)
        
    for i in range(len(cosim_matrix)):
        for j in range(len(cosim_matrix[i])):
            if cosim_matrix[i][j] == 0:
                updated_matrix[i][j] = float(float(weight_factor) * m_id[i][j])
                
    return(updated_matrix)
                
def new_dictionary(movie_title_to_id,updated_matrix):
    
    updated_dict = {}
    id_to_movie_name = {value:key for key, value in movie_title_to_id.items()}
    movies = []
    list_of_sim = []
    list_of_movies_sim = []
    inner_dicts = []
    
    for i in range(1,len(updated_matrix)+1):
        movies.append(id_to_movie_name.get(str(i)))
        
    for i in range(len(updated_matrix)):
        sim = []
        for j in range(len(updated_matrix[i])):
            sim.append(updated_matrix[i][j])
        list_of_sim.append(sim)
    
    
    for sim in range(len(list_of_sim)):
        movies_sim = []
        for movie in range(len(list_of_sim[sim])):
            movies_sim.append((movies[movie],list_of_sim[sim][movie]))
        list_of_movies_sim.append(movies_sim)
    
    for movie in range(len(list_of_movies_sim)):
        list_of_dict = []
        for name,similarity in list_of_movies_sim[movie]:
            dict_name_sim = {name : similarity}
            list_of_dict.append(dict_name_sim)
        inner_dicts.append(list_of_dict)
    
    
    for i in range(len(movies)):
        updated_dict[movies[i]] = inner_dicts[i]
    return(updated_dict)
def getRecommendedItems(prefs, movie_title_to_id, itemMatch, user, item, sim_threshold=0):
    '''
        Calculates recommendations for a given user
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- itemMatch: dictionary containing similarity matrix
        -- user: string containing name of user
        -- item: The movie we are searching for
        -- sim_threshold: minimum similarity to be considered a neighbor, default is >0
        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
    '''
    
    userRatings = prefs[user]
    
    sim_ratings = itemMatch[item]
   
    movie_names = []
    scores = 0
    totalSim = 0
    
    
    
       
    for num in range(len(sim_ratings)):
        for movie in sim_ratings[num]:
            if movie not in userRatings:
                continue
            if sim_ratings[num].get(movie) <= sim_threshold:
                continue
            
            scores += userRatings[movie]*sim_ratings[num].get(movie)
            totalSim += sim_ratings[num].get(movie)
    
    
    if totalSim != 0:
        predicted_score = (scores/totalSim)
    
    try:
        return predicted_score
    except UnboundLocalError:
        return False

def loo_cv_sim(prefs, movie_title_to_id, sim, algo, sim_matrix, sim_threshold=0):
    '''
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     Parameters:
         -- prefs dataset: critics, MLK-100k
         -- sim: distance, pearson
         -- algo: user-based (getRecommendationSim), item-based recommender (getRecommendedItems)
         -- sim_matrix: pre-computed similarity matrix
         -- sim_weighting: similarity significance weighting factor (0, 25, 50), default is 0 [None]
         -- sim_threshold: minimum similarity to be considered a neighbor, default is >0
    Returns:
         -- errors: MSE, MAE, RMSE totals for this set of conditions
         -- error_lists: MSE and MAE lists of actual-predicted differences
    '''
    errors = {}
    error_lists = {}
    mse_list = []
    mae_list = []

    pred_found = False
    recs = []

    c = 0

    # Create a temp copy of prefs
    prefs_cp = copy.deepcopy(prefs)

    for user in prefs:
        # Progress status
        c += 1
        if c % 25 == 0:
            percent_complete = (100*c)/len(prefs)
            print("%.2f %% complete" % percent_complete)

        for item in prefs[user]:
            removed_rating = prefs_cp[user].pop(item)
            #print(item)
            recs = algo(prefs_cp, movie_title_to_id, sim_matrix, user, item, sim_threshold)
            
            
            if recs != False:        
                predicted_rating = recs
                error_mse = (predicted_rating - removed_rating)**2
                error_mae = abs(predicted_rating - removed_rating)
                mse_list.append(error_mse)
                mae_list.append(error_mae)
                   
                prefs_cp[user][item] = removed_rating

    errors['mse'] = np.average(mse_list)
    errors['mae'] = np.average(mae_list)
    errors['rmse'] = np.sqrt(np.average(mse_list))

    error_lists['(r)mse'] = mse_list
    error_lists['mae'] = mae_list

    return errors, error_lists

def main():
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    print()
    prefs = {}
    done = False
    
    while not done:
        print()
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead) ml100K data from file?, \n'
                        'FE(ature Encoding) Setup?, \n'
                        'TFIDF(and cosine sim Setup)?, \n'
                        'CBR-FE(content-based recommendation Feature Encoding)?, \n'
                        'CBR-TF(content-based recommendation TF-IDF/CosineSim)? \n'
                        'LCVSIM(eave one out cross-validation)? \n'
                        '==>> '
                        )
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data' # for userids use 'critics_ratings_userIDs.data'
            itemfile = 'critics_movies.item'
            genrefile = 'critics_movies.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys())) 
            ##
            ## delete this next line when you get genres (above) working
            
            ##
            print ('Number of distinct genres: %d, number of feature profiles: %d' % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratngs file
            itemfile = 'u.item'  # movie titles file
            genrefile = 'u.genre' # movie genre file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            movies, genres, features = from_file_to_2D(path, file_dir+genrefile, file_dir+itemfile)
            ##
            ## delete this next line when you get genres (above) working
           
            ##
            print('Number of users: %d\nList of users [0:10]:' 
                  % len(prefs), list(prefs.keys())[0:10] ) 
            print ('Number of distinct genres: %d, number of feature profiles: %d' 
                   % (len(genres), len(features)))
            print('genres')
            print(genres)
            print('features')
            print(features)
            
        elif file_io == 'FE' or file_io == 'fe':
            print()
            #movie_title_to_id = movie_to_ID(movies)
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                
                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                '''      
                print('critics')
                print(R)
                print()
                print('features')
                print(features)

            elif len(prefs) > 10:
                print('ml-100k')   
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                
            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'TFIDF' or file_io == 'tfidf':
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                
                '''
                # e.g., critics data (CES)
                R = np.array([
                [2.5, 3.5, 3.0, 3.5, 2.5, 3.0],
                [3.0, 3.5, 1.5, 5.0, 3.5, 3.0],
                [2.5, 3.0, 0.0, 3.5, 0.0, 4.0],
                [0.0, 3.5, 3.0, 4.0, 2.5, 4.5],
                [3.0, 4.0, 2.0, 3.0, 2.0, 3.0],
                [3.0, 4.0, 0.0, 5.0, 3.5, 3.0],
                [0.0, 4.5, 0.0, 4.0, 1.0, 0.0],
                ])            
                '''      
                print('critics')
                print(R)
                print()
                print('features')
                print(features)
                print()
                print('feature docs')
                print(feature_docs) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print(cosim_matrix)
                user = input("Which user do you want to get ratings from: ")
                movie_title_to_id = movie_to_ID(movies)
                sim_threshold = input("Choose a Similarity Threshold (0,.340,.530,.903,1): ")
                if float(sim_threshold) > 1 or float(sim_threshold) < 0:
                    sim_threshold = input("Similarity Threshold must be between 0 and 1. Please try again: ")
                sim_threshold = float(sim_threshold)
                print(get_TFIDF_recommendations(prefs,cosim_matrix,user,movie_title_to_id,sim_threshold))
                 
                '''
                <class 'numpy.ndarray'> 
                
                [[1.         0.         0.35053494 0.         0.         0.61834884]
                [0.         1.         0.19989455 0.17522576 0.25156892 0.        ]
                [0.35053494 0.19989455 1.         0.         0.79459157 0.        ]
                [0.         0.17522576 0.         1.         0.         0.        ]
                [0.         0.25156892 0.79459157 0.         1.         0.        ]
                [0.61834884 0.         0.         0.         0.         1.        ]]
                '''
                
                


            elif len(prefs) > 10:
                print('ml-100k')
                # convert prefs dictionary into 2D list
                R = to_array(prefs)
                feature_str = to_string(features)                 
                feature_docs = to_docs(feature_str, genres)
                
                
                
                
                print(R[:3][:5])
                print()
                print('features')
                print(features[0:5])
                print()
                print('feature docs')
                print(feature_docs[0:5]) 
                cosim_matrix = cosine_sim(feature_docs)
                print()
                print('cosine sim matrix')
                print (type(cosim_matrix), len(cosim_matrix))
                print()
                print(cosim_matrix)
                user = input("Which user do you want to get ratings from: ")
                movie_title_to_id = movie_to_ID(movies)
                sim_threshold = input("Choose a Similarity Threshold (0,.340,.530,.903,1): ")
                if float(sim_threshold) > 1 or float(sim_threshold) < 0:
                    sim_threshold = input("Similarity Threshold must be between 0 and 1. Please try again: ")
                sim_threshold = float(sim_threshold)
                print(get_TFIDF_recommendations(prefs,cosim_matrix,user,movie_title_to_id,sim_threshold))
                
                '''
                <class 'numpy.ndarray'> 1682
                
                [[1.         0.         0.         ... 0.         0.34941857 0.        ]
                 [0.         1.         0.53676706 ... 0.         0.         0.        ]
                 [0.         0.53676706 1.         ... 0.         0.         0.        ]
                 [0.18860189 0.38145435 0.         ... 0.24094937 0.5397592  0.45125862]
                 [0.         0.30700538 0.57195272 ... 0.19392295 0.         0.36318585]
                 [0.         0.         0.         ... 0.53394963 0.         1.        ]]
                '''
                cosim_matrix2 = []
                for i in range(len(cosim_matrix)):
                    for j in range(len(cosim_matrix[i])):
                        if ((i != j) and cosim_matrix[i][j] != 0):
                            cosim_matrix2.append(cosim_matrix[i][j])
                
                    
                plt.hist(cosim_matrix2, bins = 10)
                
                print("All bin mean: ", np.average(cosim_matrix2))
                print("All bin stDev: ", np.std(cosim_matrix2))
                plt.show()     
                
            else:
                print ('Empty dictionary, read in some data!')
                print()

        elif file_io == 'CBR-FE' or file_io == 'cbr-fe':
            print()
            method = "FE"
            # determine the U-I matrix to use ..
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics') 
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')

            elif len(prefs) > 10:
                print('ml-100k')   
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                #print(movies)
                replace = []
                test = get_FE_recommendations(prefs, features, movie_to_ID(movies), userID)
                for elem in test:
                    temp = movies[str(elem[1])]
                    replace.append(  ((elem[0], temp)) )
                    
                print(replace)

            else:
                print ('Empty dictionary, read in some data!')
                print()
          
        elif file_io == 'CBR-TF' or file_io == 'cbr-tf':
            print()
            if len(prefs) > 0:
                ready = False  # subc command in progress

                sim_weighting = input(
                    'Enter similarity significance weighting n/(sim_weighting): 0 [None], 25, 50\n')

                if int(sim_weighting) != 25 and int(sim_weighting) != 50:
                    sim_weighting = 0
                    print(
                        'ALERT: invalid option or 0 was selected, defaulting to no weighting\n')
                else:
                    sim_weighting = int(sim_weighting)
                    print("similarity weighting set to {}".format(sim_weighting))

                # prompt for similarity thresold, if any
                sim_threshold = input("Choose a Similarity Threshold (0,.340,.530,.903,1): ")
                if float(sim_threshold) > 1 or float(sim_threshold) < 0:
                    sim_threshold = input("Similarity Threshold must be between 0 and 1. Please try again: ")
                sim_threshold = float(sim_threshold)

                sub_cmd = input(
                    'RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson?\n')

                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open("save_itemsim_distance.p", "rb"))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(
                            open("save_itemsim_pearson.p", "rb"))
                        sim_method = 'sim_pearson'

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(
                            prefs, similarity=sim_distance, sim_weighting=sim_weighting, sim_threshold=sim_threshold)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open(
                            "save_itemsim_distance.p", "wb"))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix

                        itemsim = calculateSimilarItems(
                            prefs, similarity=sim_pearson, sim_weighting=sim_weighting, sim_threshold=sim_threshold)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open(
                            "save_itemsim_pearson.p", "wb"))
                        sim_method = 'sim_pearson'

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue

                    ready = True  # sub command completed successfully

                except Exception as ex:
                    print('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                          ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()

                if len(itemsim) > 0 and ready == True and len(itemsim) <= 10:
                    # Only want to print if sub command completed successfully
                    print('Similarity matrix based on %s, len = %d'
                          % (sim_method, len(itemsim)))
                    print()
            else:
                print('Empty dictionary, R(ead) in some data!')
                return
            
            R = to_array(prefs)
            feature_str = to_string(features)                 
            feature_docs = to_docs(feature_str, genres)
            # determine the U-I matrix to use ..
            
            if len(prefs) > 0 and len(prefs) <= 10: # critics
                print('critics')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                weight_factor = input("Choose a Weight Factor (0,.25,.5,.75,1): ")
                if float(weight_factor) > 1 or float(weight_factor) < 0:
                    weight_factor = input("Weight Factor must be between 0 and 1. Please try again: ")
                cosim_matrix = cosine_sim(feature_docs)
                movie_title_to_id = movie_to_ID(movies)
                updated_matrix = update_Cosim_Matrix(movie_title_to_id,itemsim,cosim_matrix,weight_factor)
                print(get_TFIDF_recommendations(prefs,updated_matrix,userID,movie_title_to_id,sim_threshold))
            
            elif len(prefs) > 10:
                print('ml-100k')
                userID = input('Enter username (for critics) or userid (for ml-100k) or return to quit: ')
                weight_factor = input("Choose a Weight Factor (0,.25,.5,.75,1): ")
                if float(weight_factor) > 1 or float(weight_factor) < 0:
                    weight_factor = input("Weight Factor must be between 0 and 1. Please try again: ")
                cosim_matrix = cosine_sim(feature_docs)
                movie_title_to_id = movie_to_ID(movies)
                updated_matrix = update_Cosim_Matrix(movie_title_to_id,itemsim,cosim_matrix,weight_factor)
                print(get_TFIDF_recommendations(prefs,updated_matrix,userID,movie_title_to_id,sim_threshold))
                
            else:
                print ('Empty dictionary, read in some data!')
                print()
        
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            R = to_array(prefs)
            feature_str = to_string(features)                 
            feature_docs = to_docs(feature_str, genres)
            movie_title_to_id = movie_to_ID(movies)
            print()
            if method == "TF":
                
                updated_matrix = update_Cosim_Matrix(movie_title_to_id,itemsim,cosim_matrix,weight_factor)
                updated_dict = new_dictionary(movie_title_to_id,updated_matrix)
                if len(prefs) > 0 and updated_dict != {}:
                    print('LOO_CV_SIM Evaluation')

                # check for small or large dataset (for print statements)
                if len(prefs) <= 7:
                    prefs_name = 'critics'
                else:
                    prefs_name = 'MLK-100k'

                if sim_method == 'sim_pearson':
                    sim = sim_pearson
                else:
                    sim = sim_distance
                errors, error_lists = loo_cv_sim(
                    prefs, movie_title_to_id, sim, getRecommendedItems, updated_dict, sim_threshold=sim_threshold)
                print('Errors for %s: MSE = %.5f, MAE = %.5f, RMSE = %.5f, len(SE list): %d, using %s with sim_threshold >%f and sim_weighting of %s'
                      % (prefs_name, errors['mse'], errors['mae'], errors['rmse'], len(error_lists['(r)mse']), sim_method, sim_threshold, str(len(error_lists['(r)mse']))+'/' + str(sim_weighting)))
                print()

            
            
            else:
                cosim_matrix = cosine_sim(feature_docs)
                updated_dict = new_dictionary(movie_title_to_id,cosim_matrix) 
                if len(prefs) > 0 and updated_dict != {}:
                    print('LOO_CV_SIM Evaluation')

                    # check for small or large dataset (for print statements)
                    if len(prefs) <= 7:
                        prefs_name = 'critics'
                    else:
                        prefs_name = 'MLK-100k'

                    sim_method = "CBR-FE"
                    sim = "FE"
                    sim_threshold = 0
                    sim_weighting = 0
                    errors, error_lists = loo_cv_sim(
                        prefs, movie_title_to_id, sim, getRecommendedItems, updated_dict, sim_threshold)
                    print('Errors for %s: MSE = %.5f, MAE = %.5f, RMSE = %.5f, len(SE list): %d, using %s with sim_threshold >%0.1f and sim_weighting of %s'
                          % (prefs_name, errors['mse'], errors['mae'], errors['rmse'], len(error_lists['(r)mse']), sim_method, sim_threshold, str(len(error_lists['(r)mse']))+'/' + str(sim_weighting)))
                    print()


        else:
            done = True
       
    print('Goodbye!')  
    
if __name__ == "__main__":
    main()    
    
    
'''
Sample output ..


==>> cbr-fe
ml-100k

Enter username (for critics) or userid (for ml-100k) or return to quit: 340
rec for 340 = [
(5.0, 'Woman in Question, The (1950)'), 
(5.0, 'Wallace & Gromit: The Best of Aardman Animation (1996)'), 
(5.0, 'Thin Man, The (1934)'), 
(5.0, 'Maltese Falcon, The (1941)'), 
(5.0, 'Lost Highway (1997)'), 
(5.0, 'Faust (1994)'), 
(5.0, 'Daytrippers, The (1996)'), 
(5.0, 'Big Sleep, The (1946)'), 
(4.836990595611285, 'Sword in the Stone, The (1963)'), 
(4.836990595611285, 'Swan Princess, The (1994)')]

==>> cbr-tf
ml-100k

Enter username (for critics) or userid (for ml-100k) or return to quit: 340
rec for 340 =  [
(5.000000000000001, 'Wallace & Gromit: The Best of Aardman Animation (1996)'), 
(5.000000000000001, 'Faust (1994)'), 
(5.0, 'Woman in Question, The (1950)'), 
(5.0, 'Thin Man, The (1934)'), 
(5.0, 'Maltese Falcon, The (1941)'), 
(5.0, 'Lost Highway (1997)'), 
(5.0, 'Daytrippers, The (1996)'), 
(5.0, 'Big Sleep, The (1946)'), 
(4.823001861184155, 'Sword in the Stone, The (1963)'), 
(4.823001861184155, 'Swan Princess, The (1994)')]

'''

