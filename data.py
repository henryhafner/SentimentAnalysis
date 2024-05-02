from keras.datasets import imdb

training_set, testing_set = imdb.load_data(index_from = 3)

word_index=imdb.get_word_index()    #dict that maps words to ints
reversed_word_index = {index+3: word for word,index in word_index.items()}  #reverses word_index to map ints to words and accounts for index_from
reversed_word_index[0] = "[padding]"    #inserts padding
reversed_word_index[1] = "[start]"  
reversed_word_index[2] = "[oov]"

def display_entries(dataset, nums):
    # **ADD YOUR CODE HERE **
    dataset_data, dataset_labels = dataset  #split dataset into data and labels
    for num in nums:
        print("Encoded review: ",dataset_data[num]) #prints encoded int array
        english = " ".join(reversed_word_index.get(i) for i in dataset_data[num])   #translates ints to words with reverse dict
        print("English review: ",english)
        if dataset_labels[num]==1:  #checks if sentiment was positive or negative
            print("Sentiment: Positive\n")
        else:
            print("Sentiment: Negative\n")

# ** Test code here
display_entries(training_set, [6,10,159,169,45])
display_entries(testing_set, [6,10,159,169,45])
# used for presentation examples
# display_entries(training_set, [159]) 
# display_entries(testing_set, [10]) 

