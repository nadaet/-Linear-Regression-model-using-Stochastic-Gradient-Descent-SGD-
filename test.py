import numpy as np


X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)


m = 0.0  
b = 0.0  
learning_rate = 0.01
epochs = 100


for epoch in range(epochs):
    for i in range(len(X)):
        
        xi = X[i]
        yi = y[i]
        
        
        y_pred = m * xi + b
        
       
        dm = -2 * xi * (yi - y_pred)
        db = -2 * (yi - y_pred)
        
    
        m -= learning_rate * dm
        b -= learning_rate * db
    
    
    if epoch % 10 == 0:
        total_loss = np.mean((y - (m * X + b)) ** 2)
        print(f"Epoch {epoch}: m={m:.4f}, b={b:.4f}, loss={total_loss:.4f}")

print(f"\nFinal model: y = {m:.2f}x + {b:.2f}")

