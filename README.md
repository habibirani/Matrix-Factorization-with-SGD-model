# Matrix Factorization with Adaptive Learning Rate and Fractional Optimization

Code from the paper titled "Auto-tuning HyperParameters of SGD Matrix Factorization-Based Recommender Systems Using Genetic Algorithm".

Best Student Paper Award of 2022 IEEE International Conference on Omni-layer Intelligent Systems (COINS) [![Conference](https://img.shields.io/badge/Conference-2022-008000.svg)](https://coinsconf.com)

#### Auto-tuning HyperParameters of SGD Matrix Factorization-Based Recommender Systems Using Genetic Algorithm Paper: [PDF](https://ieeexplore.ieee.org/abstract/document/9854956)

This Python script performs matrix factorization using adaptive learning rate and fractional optimization techniques to predict user-item ratings in a collaborative filtering setting. The code is designed to handle the MovieLens dataset with user ratings for movies.

## Requirements
- Python 3.x
- NumPy
- Pandas
- SciPy
- scikit-learn

## Dataset
The code expects the MovieLens dataset to be available in a file named 'u.data'. The dataset should contain four columns: 'user_id', 'movie_id', 'rating', and 'timestamp'.

## Code Explanation
1. The MovieLens dataset is loaded, and the user-item rating matrix 'R' is created.
2. The 'Train' and 'Test' matrices are initialized as deep copies of 'R', with some elements set to 0 for training and testing.
3. The matrix factorization function is defined to decompose the 'Train' matrix into user and item matrices 'P' and 'Q' using gradient descent.
4. Adaptive learning rate is used to adjust the learning rate during training to improve convergence.
5. Fractional optimization is implemented to add nonlinearity to the learning process for better performance.
6. The algorithm stops when either the specified number of steps is reached or the error falls below a certain threshold.
7. The optimal 'P' and 'Q' matrices are obtained after training, and the final predicted ratings are calculated.
8. Root Mean Squared Error (RMSE) is used to evaluate the performance of the model.

## Usage
1. Ensure you have the 'u.data' file in the same directory as the script.
2. Install the required Python libraries if you haven't already.
3. Run the script to perform matrix factorization and obtain the predicted ratings.
4. The RMSE will be displayed to evaluate the model's performance.

Feel free to modify the code as needed and experiment with different hyperparameters to achieve better results. Enjoy experimenting with matrix factorization and collaborative filtering for movie recommendations!

<!-- CITATION -->
## Citation


```bibtex
@inproceedings{irani2022auto,
  title={Auto-tuning HyperParameters of SGD Matrix Factorization-Based Recommender Systems Using Genetic Algorithm},
  author={Irani, Habib and Elahi, Fatemeh and Fazlali, Mahmood and Shahsavari, Mahyar and Farahani, Bahar},
  booktitle={2022 IEEE International Conference on Omni-layer Intelligent Systems (COINS)},
  pages={1--7},
  year={2022},
  organization={IEEE}
}
```
