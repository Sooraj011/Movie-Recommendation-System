from django.shortcuts import render,HttpResponseRedirect
from django.contrib.auth import authenticate,login,logout,update_session_auth_hash
from django.contrib.auth.models import User,Group
from .forms import SignUpForm, AddMovieForm, LoginForm, AddRatingForm
from .models import Movie,Rating
from django.contrib import messages
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Create your views here.

def filterMovieByGenre():
     #filtering by genres
    allMovies=[]
    genresMovie= Movie.objects.values('genres', 'id')
    genres= {item["genres"] for item in genresMovie}
    for genre in genres:
        movie=Movie.objects.filter(genres=genre)
        print(movie)
        n = len(movie)
        nSlides = n // 4 + ceil((n / 4) - (n // 4))
        allMovies.append([movie, range(1, nSlides), nSlides])
    params={'allMovies':allMovies }
    return params


            
def generateRecommendation(request):
    # Fetch movies and ratings data from the database
    movies = Movie.objects.all()
    ratings = Rating.objects.all()

    # Initialize lists to store data for creating DataFrames
    movies_data = []
    ratings_data = []

    # Movie Data Frames
    for item in movies:
        movies_data.append([item.id, item.title, item.movieduration, item.image.url, item.genres])

    movies_df = pd.DataFrame(movies_data, columns=['movieId', 'title', 'movieduration', 'image', 'genres'])

    # Print initial state of movies_df
    print("Initial Movies DataFrame:")
    print(movies_df)

    # Rating Data Frames
    for item in ratings:
        ratings_data.append([item.user.id, item.movie, item.rating])

    rating_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'rating'])
    rating_df['userId'] = rating_df['userId'].astype(str).astype(np.int64)
    rating_df['movieId'] = rating_df['movieId'].astype(str).astype(np.int64)
    rating_df['rating'] = rating_df['rating'].astype(str).astype(np.float64)

    # Print initial state of rating_df
    print("Initial Rating DataFrame:")
    print(rating_df)

    # Check if the user is authenticated
    if request.user.is_authenticated:
        userid = request.user.id

        # select_related is a join statement in Django. It looks for a foreign key and joins the table
        user_input = Rating.objects.select_related('movie').filter(user=userid)

        # If the user hasn't rated any movies, return None
        if user_input.count() == 0:
            recommender_query = None
            user_input = None
        else:
            watched_movies_data = []

            # Collect data for movies watched by the user
            for item in user_input:
                watched_movies_data.append([item.movie.title, item.rating])

            input_movies = pd.DataFrame(watched_movies_data, columns=['title', 'rating'])
            input_movies['rating'] = input_movies['rating'].astype(str).astype(np.float64)

            # Print watched movies by the user DataFrame
            print("Watched Movies by User DataFrame:")
            print(input_movies)

            # Filtering out the movies by title
            input_id = movies_df[movies_df['title'].isin(input_movies['title'].tolist())]
            # Then merging it so we can get the movieId. It's implicitly merging it by title.
            input_movies = pd.merge(input_id, input_movies)

            # Print merged DataFrame after filtering movies by title
            print("Merged DataFrame after Filtering by Title:")
            print(input_movies)

            # Filtering out users that have watched movies that the input has watched and storing it
            user_subset = rating_df[rating_df['movieId'].isin(input_movies['movieId'].tolist())]

            # Print userSubset DataFrame after filtering users that have watched common movies
            print("User Subset DataFrame:")
            print(user_subset.head())

            # Groupby creates several sub-dataframes where they all have the same value in the column specified as the parameter
            user_subset_group = user_subset.groupby(['userId'])

            # Sorting it so users with movie most in common with the input will have priority
            user_subset_group = sorted(user_subset_group, key=lambda x: len(x[1]), reverse=True)

            # Print userSubsetGroup
            print("User Subset Group:")
            print(user_subset_group)

            user_subset_group = user_subset_group[0:]

            # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
            pearson_correlation_dict = {}

            # For every user group in our subset
            for name, group in user_subset_group:
                # Let's start by sorting the input and current user group so the values aren't mixed up later on
                group = group.sort_values(by='movieId')
                input_movies = input_movies.sort_values(by='movieId')

                # Get the N for the formula
                n_ratings = len(group)

                # Get the review scores for the movies that they both have in common
                temp_df = input_movies[input_movies['movieId'].isin(group['movieId'].tolist())]

                # And then store them in a temporary buffer variable in a list format to facilitate future calculations
                temp_rating_list = temp_df['rating'].tolist()

                # Let's also put the current user group reviews in a list format
                temp_group_list = group['rating'].tolist()

                # Now let's calculate the Pearson correlation between two users, so-called, x and y
                Sxx = sum([i**2 for i in temp_rating_list]) - pow(sum(temp_rating_list), 2) / float(n_ratings)
                Syy = sum([i**2 for i in temp_group_list]) - pow(sum(temp_group_list), 2) / float(n_ratings)
                Sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - sum(
                    temp_rating_list) * sum(temp_group_list) / float(n_ratings)

                # If the denominator is different than zero, then divide, else, 0 correlation.
                if Sxx != 0 and Syy != 0:
                    pearson_correlation_dict[name] = Sxy / sqrt(Sxx * Syy)
                else:
                    pearson_correlation_dict[name] = 0

            # Print Pearson Correlation Dictionary
            print("Pearson Correlation Dictionary:")
            print(pearson_correlation_dict.items())

            pearson_df = pd.DataFrame.from_dict(pearson_correlation_dict, orient='index')
            pearson_df.columns = ['similarityIndex']
            pearson_df['userId'] = pearson_df.index
            pearson_df.index = range(len(pearson_df))

            # Print Pearson DataFrame
            print("Pearson DataFrame:")
            print(pearson_df.head())

            top_users = pearson_df.sort_values(by='similarityIndex', ascending=False)[0:]

            # Print Top Users DataFrame
            print("Top Users DataFrame:")
            print(top_users.head())

            top_users['userId'] = top_users['userId'].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None)
            #pd.to_numeric(top_users['userId'], errors='coerce').astype('Int64')
            
            # Print top_users DataFrame after userId conversion
            print("Top Users DataFrame after UserId Conversion:")
            print(top_users)


            top_users_rating = top_users.merge(rating_df, left_on='userId', right_on='userId', how='inner')

            # Print Top Users Rating DataFrame
            print("Top Users Rating DataFrame:")
            print(top_users_rating.head())

            # Multiplies the similarity by the user's ratings
            top_users_rating['weightedRating'] = top_users_rating['similarityIndex'] * top_users_rating['rating']

            # Print weighted Rating DataFrame
            print("Weighted Rating DataFrame:")
            print(top_users_rating.head())

            # Applies a sum to the topUsers after grouping it up by userId
            temp_top_users_rating = top_users_rating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
            temp_top_users_rating.columns = ['sum_similarityIndex', 'sum_weightedRating']

            # Print temporary Top Users Rating DataFrame
            print("Temporary Top Users Rating DataFrame:")
            print(temp_top_users_rating.head())

            # Creates an empty dataframe
            recommendation_df = pd.DataFrame()

            # Now we take the weighted average
            recommendation_df['weighted average recommendation score'] = temp_top_users_rating[
                                                                           'sum_weightedRating'] / temp_top_users_rating[
                                                                           'sum_similarityIndex']

            recommendation_df['movieId'] = temp_top_users_rating.index

            # Print Recommendations DataFrame
            print("Recommendations DataFrame:")
            print(recommendation_df.head())

            recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

            recommender = movies_df.loc[movies_df['movieId'].isin(
                recommendation_df.head(5)['movieId'].tolist())]

            # Print Recommended Movies
            print("Recommended Movies:")
            print(recommender)
            print("________________________________________________________________________")

            # Return recommended movies as a list of records
            return recommender.to_dict('records')

            


    

def signup(request):
    if not request.user.is_authenticated:
        if request.method=='POST':
            fm=SignUpForm(request.POST)
            if fm.is_valid():
                user=fm.save()
                group, created=Group.objects.get_or_create(name='Editor')
                user.groups.add(group)
                messages.success(request,'Account Created Successfully!!!')
        else:
            if not request.user.is_authenticated:
                fm=SignUpForm()
        return render(request,'recommender/signup.html',{'form':fm})
    else:
        return HttpResponseRedirect('/home/')

def user_login(request):
    if not request.user.is_authenticated:
        if request.method=='POST':
            fm=LoginForm(request=request,data=request.POST)
            if fm.is_valid():
                uname=fm.cleaned_data['username']
                upass=fm.cleaned_data['password']
                user=authenticate(username=uname,password=upass)
                if user is not None:
                    login(request,user)
                    messages.success(request,'Logged in Successfully!!')
                    return HttpResponseRedirect('/dashboard/')
        else:
            fm=LoginForm()
        return render(request,'recommender/login.html',{'form':fm})
    else:
        return HttpResponseRedirect('/dashboard/')



def home(request):
    params=filterMovieByGenre()
    params['recommended']=generateRecommendation(request)
    print("recomended:",params['recommended'])
    return render(request,'recommender/home.html',params)

def addmovie(request):
    if request.user.is_authenticated:
        if request.method=='POST':
            fm=AddMovieForm(request.POST,request.FILES)
            if fm.is_valid():
                fm.save()
                messages.success(request,'Movie Added Successfully!!!')
        else:
            fm=AddMovieForm()
        return render(request,'recommender/addmovie.html',{'form':fm})
    else:
        return HttpResponseRedirect('/login/')


def dashboard(request):
    if request.user.is_authenticated: 
        params=filterMovieByGenre()
        params['user']=request.user
        if request.method=='POST':
            userid=request.POST.get('userid')
            movieid=request.POST.get('movieid')
            movie=Movie.objects.all()
            u=User.objects.get(pk=userid)
            m=Movie.objects.get(pk=movieid)
            rfm=AddRatingForm(request.POST)
            params['rform']=rfm
            if rfm.is_valid():
                rat=rfm.cleaned_data['rating']
                count=Rating.objects.filter(user=u,movie=m).count()
                if(count>0):
                    messages.warning(request,'You have already submitted your review!!')
                    return render(request,'recommender/dashboard.html',params)
                action=Rating(user=u,movie=m,rating=rat)
                action.save()
                messages.success(request,'You have submitted'+' '+rat+' '+"star")
            return render(request,'recommender/dashboard.html',params)
        else:
            #print(request.user.id)
            rfm=AddRatingForm()
            params['rform']=rfm
            movie=Movie.objects.all()
            return render(request,'recommender/dashboard.html',params)
    else:
        return HttpResponseRedirect('/login/')
            
def user_logout(request):
    if request.user.is_authenticated:
        logout(request)
        return HttpResponseRedirect('/login/')


def profile(request):
    if request.user.is_authenticated:
        #"select sum(rating) from Rating where user=request.user.id"
        r=Rating.objects.filter(user=request.user.id)
        totalReview=0
        for item in r:
            totalReview+=int(item.rating)
        #select count(*) from Rating where user=request.user.id"
        totalwatchedmovie=Rating.objects.filter(user=request.user.id).count()
        return render(request,'recommender/profile.html',{'totalReview':totalReview,'totalwatchedmovie':totalwatchedmovie})
    else:
        return HttpResponseRedirect('/login/')
    
    
    
    
    

# def generateRecommendation(request):
#     movie=Movie.objects.all()
#     rating=Rating.objects.all()
#     x=[] 
#     y=[]
#     A=[]
#     B=[]
#     C=[]
#     D=[]
#     #Movie Data Frames
#     for item in movie:
#         x=[item.id,item.title,item.movieduration,item.image.url,item.genres] 
#         y+=[x]
#     movies_df = pd.DataFrame(y,columns=['movieId','title','movieduration','image','genres'])
#     print("Movies DataFrame")
#     print(movies_df)
#     print(movies_df.dtypes)
#     #Rating Data Frames
#     print(rating)
#     for item in rating:
#         A=[item.user.id,item.movie,item.rating]
#         B+=[A]
#     rating_df=pd.DataFrame(B,columns=['userId','movieId','rating'])
#     print("Rating data Frame")
#     rating_df['userId']=rating_df['userId'].astype(str).astype(np.int64)
#     rating_df['movieId']=rating_df['movieId'].astype(str).astype(np.int64)
#     rating_df['rating']=rating_df['rating'].astype(str).astype(np.float64)
#     print(rating_df)
#     print(rating_df.dtypes)
#     if request.user.is_authenticated:
#         userid=request.user.id
#         #select related is join statement in django.It looks for foreign key and join the table
#         userInput=Rating.objects.select_related('movie').filter(user=userid)
#         if userInput.count()== 0:
#             recommenderQuery=None
#             userInput=None
#         else:
#             for item in userInput:
#                 C=[item.movie.title,item.rating]
#                 D+=[C]
#             inputMovies=pd.DataFrame(D,columns=['title','rating'])
#             print("Watched Movies by user dataframe")
#             inputMovies['rating']=inputMovies['rating'].astype(str).astype(np.float64)
#             print(inputMovies.dtypes)

#             #Filtering out the movies by title
#             inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#             #Then merging it so we can get the movieId. It's implicitly merging it by title.
#             inputMovies = pd.merge(inputId, inputMovies)
#             # #Dropping information we won't use from the input dataframe
#             # inputMovies = inputMovies.drop('year', 1)
#             #Final input dataframe
#             #If a movie you added in above isn't here, then it might not be in the original 
#             #dataframe or it might spelled differently, please check capitalisation.
#             print("<--------------------------------------------->")
#             print(inputMovies)
#             print("<--------------------------------------------->")

#             #Filtering out users that have watched movies that the input has watched and storing it
#             userSubset = rating_df[rating_df['movieId'].isin(inputMovies['movieId'].tolist())]
#             print("<--------------------------------------------->")
#             print(userSubset.head())
#             print("<--------------------------------------------->")

#             #Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
#             userSubsetGroup = userSubset.groupby(['userId'])
            
#             #print(userSubsetGroup.get_group(7))

#             #Sorting it so users with movie most in common with the input will have priority
#             userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
#             print("{------------------------------}")
#             print(userSubsetGroup)
#             print("{------------------------------}")


#             userSubsetGroup = userSubsetGroup[0:]


#             #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
#             pearsonCorrelationDict = {}

#         #For every user group in our subset
#             for name, group in userSubsetGroup:
#             #Let's start by sorting the input and current user group so the values aren't mixed up later on
#                 group = group.sort_values(by='movieId')
#                 print("{-----------------------}\n",group,"{------------------------------}\n")
#                 inputMovies = inputMovies.sort_values(by='movieId')
#                 #Get the N for the formula
#                 nRatings = len(group)
#                 #Get the review scores for the movies that they both have in common
#                 temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
#                 #And then store them in a temporary buffer variable in a list format to facilitate future calculations
#                 tempRatingList = temp_df['rating'].tolist()
#                 #Let's also put the current user group reviews in a list format
#                 tempGroupList = group['rating'].tolist()
#                 #Now let's calculate the pearson correlation between two users, so called, x and y
#                 Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
#                 Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
#                 Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
                
#                 #If the denominator is different than zero, then divide, else, 0 correlation.
#                 if Sxx != 0 and Syy != 0:
#                     pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
#                 else:
#                     pearsonCorrelationDict[name] = 0

#             print(pearsonCorrelationDict.items())

#             pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
#             pearsonDF.columns = ['similarityIndex']
#             pearsonDF['userId'] = pearsonDF.index
#             pearsonDF.index = range(len(pearsonDF))
#             print(pearsonDF.head())

#             topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:]
#             print(topUsers.head())

#             topUsers['userId'] = pd.to_numeric(topUsers['userId'], errors='coerce').astype('Int64')

#             topUsersRating=topUsers.merge(rating_df, left_on='userId', right_on='userId', how='inner')
#             topUsersRating.head()

#                 #Multiplies the similarity by the user's ratings
#             topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
#             topUsersRating.head()


#             #Applies a sum to the topUsers after grouping it up by userId
#             tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
#             tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
#             tempTopUsersRating.head()

#             #Creates an empty dataframe
#             recommendation_df = pd.DataFrame()
#             #Now we take the weighted average
#             recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
#             recommendation_df['movieId'] = tempTopUsersRating.index
#             recommendation_df.head()

#             recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
#             recommender=movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(5)['movieId'].tolist())]
#             print(recommender)
#             print("__________________________________________________________________________________________________________________________________________________________________")
#             return recommender.to_dict('records')