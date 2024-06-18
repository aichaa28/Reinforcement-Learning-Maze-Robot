import tkinter as tk
import numpy as np

# Définition du labyrinthe
maze = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  0,  0,  0, -1,  0,  0,  0,  0, -1],
    [-1,  0, -1,  0,  0,  0, -1,  0, -1, -1],
    [-1,  0, -1,  0, -1, -1, -1,  0,  0, -1],
    [-1,  0,  0,  0, -1,  0,  0,  0,  0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
]

# Paramètres de Q-learning
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

# Actions possibles : haut, bas, gauche, droite
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Initialisation de la matrice Q avec des zéros
Q = np.zeros((len(maze), len(maze[0]), len(actions)))

# Fonction pour choisir une action selon la politique 𝜀-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(Q[state[0]][state[1]])

# Fonction pour mettre à jour la matrice Q
def update_Q(state, action, reward, next_state):
    Q[state[0]][state[1]][action] = (1 - gamma) * Q[state[0]][state[1]][action] + gamma * (reward + np.max(Q[next_state[0]][next_state[1]]))

# Algorithme de Q-learning
for _ in range(num_episodes):
    state = (1, 1)  # Position initiale
    while True:
        action = choose_action(state)
        new_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        if maze[new_state[0]][new_state[1]] == -1:  # Si la prochaine action est un mur, réinitialiser la position
            new_state = state
        reward = maze[new_state[0]][new_state[1]]
        update_Q(state, action, reward, new_state)
        state = new_state
        if maze[state[0]][state[1]] == 100:  # Si la sortie est atteinte, terminer l'épisode
            break

# Trouver le chemin optimal
path = []
state = (1, 1)  # Position initiale
while maze[state[0]][state[1]] != 100:
    action = np.argmax(Q[state[0]][state[1]])
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    path.append(new_state)
    state = new_state

# Affichage graphique avec Tkinter
root = tk.Tk()
root.title("Labyrinthe")

canvas = tk.Canvas(root, width=600, height=400, bg='white')
canvas.pack()

# Dessiner le labyrinthe
for i in range(len(maze)):
    for j in range(len(maze[0])):
        x0, y0 = j * 40, i * 40
        x1, y1 = x0 + 40, y0 + 40
        color = 'white' if maze[i][j] == 0 else 'black'
        canvas.create_rectangle(x0, y0, x1, y1, fill=color)

# Dessiner le chemin optimal
for pos in path:
    x0, y0 = pos[1] * 40 + 5, pos[0] * 40 + 5
    x1, y1 = x0 + 30, y0 + 30
    canvas.create_oval(x0, y0, x1, y1, fill='green')

root.mainloop()