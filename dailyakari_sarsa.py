import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import random

# === パズル定義 ===
rows, cols = 5, 5
grid = np.zeros((rows, cols), dtype=int)
numbered_walls = {(0,1):0, (1,3):1, (3,1):2}
plain_walls = []
for (r, c), v in numbered_walls.items(): grid[r, c] = 10
for r, c in plain_walls: grid[r, c] = -1
original_numbers = numbered_walls.copy()

# === 光計算・残り電球 ===
def compute_shine(lights):
    shines = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if lights[r, c] == 2:
                shines[r, c] = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    while 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == -1 or grid[rr, cc] == 10: break
                        shines[rr, cc] = True
                        rr += dr; cc += dc
    return shines

def remaining_number(lights):
    remain = {}
    for (r,c), v in original_numbers.items():
        cnt = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if 0<=rr<rows and 0<=cc<cols and lights[rr,cc]==2: cnt+=1
        remain[(r,c)] = max(v-cnt,0)
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
                        if grid[rr,cc]==-1 or grid[rr,cc]==10: break
                        if lights[rr,cc]==2: return False
                        rr+=dr; cc+=dc
    remain = remaining_number(lights)
    if any(v>0 for v in remain.values()): return False
    return True

# --- 状態ベクトル ---
def get_state(lights): return tuple(lights.flatten())

# --- SARSA 報酬 ---
def get_reward(lights, action):
    temp = lights.copy()
    temp[action]=2
    reward = np.sum(compute_shine(temp))
    remain = remaining_number(temp)
    for v in remain.values(): 
        if v<0: reward-=5
    if check_clear(temp): reward+=100
    return reward

# --- SARSA 学習 ---
Q = {}
alpha=0.1; gamma=0.9; epsilon=0.3; episodes=1600

for ep in range(episodes):
    lights = np.zeros_like(grid)
    state = get_state(lights)
    # 初期行動選択
    shines = compute_shine(lights)
    actions = [(r,c) for r in range(rows) for c in range(cols)
               if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
               and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                       for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
    if not actions: continue
    if random.random()<epsilon: action=random.choice(actions)
    else: action=max({a:Q.get(state,{}).get(a,0) for a in actions}, key=lambda x:{a:Q.get(state,{}).get(a,0) for a in actions}[x])

    for step in range(25):
        reward = get_reward(lights, action)
        lights[action]=2
        next_state=get_state(lights)
        # 次の行動選択（SARSAの特徴）
        shines = compute_shine(lights)
        next_actions = [(r,c) for r in range(rows) for c in range(cols)
                        if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
                        and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
        if not next_actions:
            next_q = 0
            next_action=None
        else:
            if random.random()<epsilon: next_action=random.choice(next_actions)
            else: next_action=max({a:Q.get(next_state,{}).get(a,0) for a in next_actions}, key=lambda x:{a:Q.get(next_state,{}).get(a,0) for a in next_actions}[x])
            next_q = Q.get(next_state,{}).get(next_action,0)
        # Q値更新
        if state not in Q: Q[state]={}
        Q[state][action] = Q[state].get(action,0)+alpha*(reward + gamma*next_q - Q[state].get(action,0))
        if next_action is None or reward>=100: break
        state, action = next_state, next_action

# --- 学習結果で解を作る ---
lights = np.zeros_like(grid)
history=[lights.copy()]
for _ in range(25):
    state = get_state(lights)
    shines = compute_shine(lights)
    actions = [(r,c) for r in range(rows) for c in range(cols)
               if grid[r,c]==0 and lights[r,c]==0 and not shines[r,c]
               and all(remaining_number(lights).get((r+dr,c+dc),0)>0
                       for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0<=r+dr<rows and 0<=c+dc<cols and grid[r+dr,c+dc]==10)]
    if not actions: break
    q_vals={a:Q.get(state,{}).get(a,0) for a in actions}
    action=max(q_vals,key=lambda x:q_vals[x])
    lights[action]=2
    history.append(lights.copy())
    if check_clear(lights): break

# === 描画・アニメーション ===
fig, ax = plt.subplots()
rects=[]
for r in range(rows):
    row_rects=[]
    for c in range(cols):
        cell=grid[r,c]
        color="black" if cell==-1 or cell==10 else "white"
        rect=plt.Rectangle((c-0.5,r-0.5),1,1,color=color)
        ax.add_patch(rect)
        row_rects.append(rect)
    rects.append(row_rects)

dots=[]
texts=[]
def update(frame):
    lights = history[frame]
    shines = compute_shine(lights)
    remain = remaining_number(lights)
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
        text = "😄" if v==0 else str(v)
        texts.append(ax.text(c,r,text,color="white",ha="center",va="center",fontsize=14,weight="bold"))
    return rects+dots+texts

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500, repeat=False)
plt.show()

# --- 初期盤面のQ値ヒートマップ ---
initial_state=get_state(np.zeros_like(grid))
action_q=Q.get(initial_state,{})
q_grid=np.full((rows,cols),np.nan)
for (r,c),val in action_q.items(): q_grid[r,c]=val

plt.figure(figsize=(5,5))
plt.imshow(q_grid,cmap="coolwarm")
for r in range(rows):
    for c in range(cols):
        if not np.isnan(q_grid[r,c]): plt.text(c,r,f"{q_grid[r,c]:.1f}",ha="center",va="center",color="black")
plt.colorbar(label="Q value")
plt.title("SARSA Q values for initial state")
plt.show()
