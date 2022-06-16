import matplotlib.pyplot as plt 
import numpy as np

def plot_scores(scores):
    plt.figure(figsize=(8, 6))
    plt.title("Scores over the episodes")
    average = np.mean(scores)
    plt.plot(scores)
    plt.axhline(y = average, color = 'r', linestyle = 'dashed')
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.show()
