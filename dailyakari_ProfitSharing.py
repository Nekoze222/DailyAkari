import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import random

# === パズル定義 ===
rows, cols = 5, 5
grid = np.zeros((rows, cols), dtype=int)
numbered_walls = {(0,1):0, (1,3):1, (3,1):2}
for (r, c), v in numbered_walls.items(): grid[r, c] = 10
original_numbers = numbered_walls.copy()

def compute_shine(lights):
    shines = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if lights[r,c]==2:
                shines[r,c]=True
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr,cc=r+dr,c+dc
                    while 0<=rr<rows and 0<=cc<cols:
                        if grid[rr,cc]==10: break
                        shines[rr,cc]=True
                        rr+=dr; cc+=dc
    return shines

def remaining_number(lights):
    remain={}
    for (r,c),v in original_numbers.items():
        cnt=0
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr,cc=r+dr,c+dc
            if 0<=rr<rows and 0<=cc<cols and lights[rr,cc]==2: cnt+=1
        remain[(r,c)]=max(v-cnt,0)
    return remain

def check_clear(lights):
    shines = compute_shine(lights)
    for r in range(rows):
        for c in range(cols):
            if grid[r,c]==0 and not shines[r,c]: return False
    for r in range(rows):
        for c in range(cols):
            if lights[r,c]==2:
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr,cc=r+dr,c+dc
                    while 0<=rr<rows and 0<=cc<cols:
                        if grid[rr,cc]==10: break
                        if lights[rr,cc]==2: return False
                        rr+=dr; cc+=dc
    remain = remaining_number(lights)
    if any(v>0 for v in remain.values()): return False
    return True

# === Profit Sharing 学習 ===
Q={}
alpha=0.2
gamma=0.9
epsilon=0.3
episodes=500

for ep in range(episodes):
    lights = np.zeros_like(grid)
    history=[]
    for step in range(25):
        shines=compute_shine(lights)
        actions=[(r,c) for r in range(rows) for c in range(cols)
                 if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
                 and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
        if not actions: break
        if random.random()<epsilon: action=random.choice(actions)
        else:
            q_vals={a:Q.get(tuple(lights.flatten()),{}).get(a,0) for a in actions}
            action=max(q_vals,key=lambda x:q_vals[x])
        lights[action]=2
        history.append((tuple(lights.flatten()), action))
        if check_clear(lights): break

    # Profit Sharing 報酬分配
    reward = 100 if check_clear(lights) else 0
    for i, (state, action) in enumerate(reversed(history)):
        if state not in Q: Q[state]={}
        Q[state][action]=Q[state].get(action,0)+alpha*(reward * (gamma**i) - Q[state].get(action,0))

# --- 学習結果で解を生成 ---
lights = np.zeros_like(grid)
history_states=[lights.copy()]
for _ in range(25):
    state=tuple(lights.flatten())
    shines=compute_shine(lights)
    actions=[(r,c) for r in range(rows) for c in range(cols)
             if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
             and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
    if not actions: break
    q_vals={a:Q.get(state,{}).get(a,0) for a in actions}
    action=max(q_vals,key=lambda x:q_vals[x])
    lights[action]=2
    history_states.append(lights.copy())
    if check_clear(lights): break

# --- アニメーション ---
fig, ax = plt.subplots()
rects=[]
for r in range(rows):
    row_rects=[]
    for c in range(cols):
        color="black" if grid[r,c]==10 else "white"
        rect=plt.Rectangle((c-0.5,r-0.5),1,1,color=color)
        ax.add_patch(rect)
        row_rects.append(rect)
    rects.append(row_rects)

dots=[]
texts=[]
def update(frame):
    lights=history_states[frame]
    shines=compute_shine(lights)
    remain=remaining_number(lights)
    for r in range(rows):
        for c in range(cols):
            if grid[r,c]==0: rects[r][c].set_facecolor("yellow" if shines[r,c] else "white")
    for dot in dots: dot.remove()
    dots.clear()
    for r in range(rows):
        for c in range(cols):
            if lights[r,c]==2: dots.append(ax.plot(c,r,"o",color="orange",markersize=18)[0])
    for t in texts: t.remove()
    texts.clear()
    for (r,c),v in remain.items():
        text="😄" if v==0 else str(v)
        texts.append(ax.text(c,r,text,color="white",ha="center",va="center",fontsize=14,weight="bold"))
    return rects+dots+texts

ani = animation.FuncAnimation(fig, update, frames=len(history_states), interval=500, repeat=False)
plt.show()
