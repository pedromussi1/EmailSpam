
<h1 align="center">Email Classification System</h1>



<a href="https://www.youtube.com/watch?v=9N2W9QBaXfw">
<kbd><img src="https://i.imgur.com/o33BFMI.png?1" alt="PernTodo Website"></kbd>
</a>

<h2>Description</h2>

<p>
1. Introduction
The goal of this project is to develop an email classification system that can automatically classify emails as either "spam" or "ham" (non-spam). This system will help users manage their email inbox more efficiently by automatically filtering out unwanted spam emails.</p>

<p>
2. Problem Statement
Email spam continues to be a significant issue, cluttering inboxes and potentially leading to security threats. The main challenge is to accurately distinguish between legitimate emails (ham) and unsolicited spam emails. Therefore, the objective is to build a machine learning model that can effectively classify emails based on their content.
</p>

<p>
3. Data Description
The dataset used for this project consists of labeled email data with two classes: "spam" and "ham". Each email is represented as a text message. The dataset includes features such as the email message and its corresponding category (spam or ham).
</p>

<p>
4. Methodology

<p>
4.1 Data Preprocessing
Missing values were handled by replacing them with empty strings.
Text data preprocessing involved converting text to lowercase, removing punctuation, and removing stop words using NLTK (Natural Language Toolkit).
The dataset was split into training and testing sets using a 80-20 split ratio.
</p>

<p>
4.2 Model Training
Text data was transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
A logistic regression model was chosen for its simplicity and effectiveness in text classification tasks.
The model was trained using the training data, consisting of the TF-IDF transformed features and corresponding target labels.
</p>

<p>
4.3 Model Evaluation
The trained model's performance was evaluated using various evaluation metrics, including accuracy, precision, recall, and F1-score.
Additionally, a confusion matrix was plotted using seaborn to visualize the model's performance in terms of true positives, false positives, true negatives, and false negatives.
</p>

<p>
5. Results
The logistic regression model achieved an accuracy of X% on the training data and Y% on the test data.
The classification report provided insights into the model's performance across different classes, including precision, recall, and F1-score.
The confusion matrix visualization helped to identify the model's strengths and weaknesses in classifying spam and ham emails.
</p>

<p>
6. Future Considerations
Hyperparameter tuning: Experiment with different hyperparameters of the logistic regression model to improve its performance further.
Feature engineering: Explore additional text preprocessing techniques and feature extraction methods to enhance the model's understanding of the email content.
Model deployment: Deploy the trained model as a service to classify emails in real-time and integrate it into existing email systems for practical use.
Continuous monitoring: Implement monitoring mechanisms to track the model's performance over time and adapt to changes in email patterns and spam tactics.
</p>

<p>
7. Conclusion
In conclusion, the email classification system developed in this project demonstrates the effectiveness of machine learning in automating the classification of spam and ham emails. By preprocessing the text data, training a logistic regression model, and evaluating its performance using various metrics, we have created a reliable system for email classification. Further improvements and future considerations will continue to enhance the system's accuracy and usability in managing email communication.
</p>

<h2>Languages and Utilities Used</h2>

<ul>
  <li><b>Python</b></li>
  <li><b>Pandas</b></li>
  <li><b>scikit-learn (sklearn)</b></li>
  <li><b>NLTK (Natural Language Toolkit)</b></li>
  <li><b>Matplotlib</b></li>
  <li><b>Seaborn</b></li>
  <li><b>NumPy</b></li>
</ul>

<h2>Environments Used</h2>

<ul>
  <li><b>Windows 11</b></li>
  <li><b>Jupyter Notebook</b></li>
</ul>

<h2>
<a href="https://github.com/pedromussi1/PernTodo/blob/main/READCODE.md">Code Breakdown Here!</a>
</h2>


<h2>Project Walk-through</h2>

<p>Download the files and save them on the same directory. Open the email_spam.ipynb on a software such as Jupyter Notebook and Run All. Make sure to install in your machine all the necessary libraries. </p>

<h3>Adding Items to the List</h3>

<p align="center">
  <kbd><img src="https://i.imgur.com/6yWAVK2.png" alt="AddingItem"></kbd>
</p>

<p>The user can write the name of a new item they want to add to the list and click on "add" to add that item. In this example I chose to add "Need to buy groceries" to the list. You can see on the second image below that the item has been added to the list.</p>

<p align="center">
  <kbd><img src="https://i.imgur.com/EffXnKy.png" alt="AddingItem1"></kbd>
</p>

<h3>Deleting an item from the list</h3>

<p align="center">
  <kbd><img src="https://i.imgur.com/Kuhy6Hr.png" alt="DeletingItem"></kbd>
</p>

<p>In case the user wants to delete an item for the list, they can simply click on the button "delete" next to the item they want to delete. In this example, I chose to delete the item "wash the dishes".</p>

<h3>Editing an item on the list</h3>

<p align="center">
  <kbd><img src="https://i.imgur.com/NEnmtvj.png" alt="Level 2"></kbd>
</p>

<p>In case the user wants to edit an item on the list, they can press the button "edit" next to the item they want to edit. In this example I chose to edit the "I need to clean my car" item on the list and change it to "I need to buy toothpaste".</p>

<p align="center">
  <kbd><img src="https://i.imgur.com/8orPkgC.png" alt="Level 3"></kbd>
</p>

<p align="center">
  <kbd><img src="https://i.imgur.com/ULVNaP3.png" alt="Level 4"></kbd>
</p>

<p>You can see now that the item has been changed and it is displayed correctly on the list.</p>
