# Step 1: Install pandas
# Open the terminal and run:
# pip install pandas

# Step 2: Read the CSV file
import pandas as pd

# Local filepath
file_path = './SpamDetection.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Step 3: Create a dictionary
my_dict = {}
# Read each line of the data. For each line, make the data the key in a dictionary, and make the target the value
# The dictionary will look like this: {data: target}
for index, row in df.iterrows():
    my_dict[row['data']] = row['Target']

# Task 1: Load the dataset and split into training and testing sets 
# (first 20 into training and the rest into testing)
training_set = {}
testing_set = {}
count = 0
for key, value in my_dict.items():
    if count < 20:
        training_set[key] = value
    else:
        testing_set[key] = value
    count += 1

# Task 2: Compute the prior probabilities: P(spam) and P(ham)
set_size = len(training_set)
spam_count = 0
ham_count = 0

for key, value in training_set.items():
    if value == 'spam':
        spam_count += 1
    else:
        ham_count += 1

p_spam = spam_count / set_size
p_ham = ham_count / set_size

print('P(spam):', p_spam)
print('P(ham):', p_ham)

# Task 3: Compute the conditional probabilities P(sentence/spam) and P(sentence/ham)
# Split the sentences into words
spam_words = {}
ham_words = {}
for key, value in training_set.items():
    words = key.split()
    for word in words:
        word = word.lower()
        # If the word is spam, increment the count of the word in the spam_words dictionary
        if value == 'spam':
            if word in spam_words:
                spam_words[word] += 1
            else:
                spam_words[word] = 1
        else:
        # If the word is ham, increment the count of the word in the ham_words dictionary
            if word in ham_words:
                ham_words[word] += 1
            else:
                ham_words[word] = 1

# Compute the total number of words in spam and ham
total_spam_words = 0
total_ham_words = 0
for key, value in spam_words.items():
    total_spam_words += value
for key, value in ham_words.items():
    total_ham_words += value
# Apply Laplace smoothing by adding number of unique words to the total number of words
total_ham_words = total_ham_words + len(ham_words) + len(spam_words)
total_spam_words = total_spam_words + len(ham_words) + len(spam_words)



# Compute the conditional probabilities
def compute_conditional_probabilities_ham(word, total_words, ham_words):
    # Apply Laplace smoothing by adding 1 to the count of the word
    return (ham_words.get(word, 0) + 1) / total_words

def compute_conditional_probabilities_spam(word, total_words, spam_words):
    # Apply Laplace smoothing by adding 1 to the count of the word
    return (spam_words.get(word, 0) + 1) / total_words


# Task 4: Compute the posterior probabilities P(spam/sentence) and P(ham/sentence)
# 2 dictionaries to store the posterior probabilities and the details of the probabilities
posterior_probabilities = {}
probabilities_details = {}

for key, value in testing_set.items():
    words = key.split()
    # Set the initial posterior probabilities to the prior probabilities
    p_spam_sentence = p_spam
    p_ham_sentence = p_ham
    for word in words:
        word = word.lower()
        # Compute the posterior probabilities for each word in the sentence
        p_spam_sentence *= compute_conditional_probabilities_spam(word, total_spam_words, spam_words)
        p_ham_sentence *= compute_conditional_probabilities_ham(word, total_ham_words, ham_words)
    # Store the posterior probabilities and the details of the probabilities
    if p_spam_sentence > p_ham_sentence:
        posterior_probabilities[key] = 'spam'
    else:
        posterior_probabilities[key] = 'ham'
    probabilities_details[key] = (p_spam_sentence, p_ham_sentence)


        
# Display the results
for sentence, classification in posterior_probabilities.items():
    p_spam_sentence, p_ham_sentence = probabilities_details[sentence]
    print(f"Sentence: '{sentence}'")
    print(f"Class: {classification}")
    print(f"P(spam/sentence): {p_spam_sentence}")
    print(f"P(ham/sentence): {p_ham_sentence}")
    print()

# Task 5: Report the accuracy of the test set
correct_count = 0
for key, value in testing_set.items():
    if posterior_probabilities[key] == value:
        correct_count += 1
accuracy = correct_count / len(testing_set)
print('Correct count:', correct_count)
print('Total count:', len(testing_set))
print('Accuracy:', accuracy)
