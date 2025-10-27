import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# === パズル定義 ===
rows, cols = 7, 7
grid = np.zeros((rows, cols), dtype=int)  # 0:白, -1:黒, 10:数字付き黒
numbered_walls = {(1,1):1, (1,5):2, (3,3):0, (5,1):3, (5,5):1}
plain_walls = [(2,4), (4,2)]

# 壁を配置
for (r, c), v in numbered_walls.items():
    grid[r, c] = 10
for r, c in plain_walls:
    grid[r, c] = -1

# 元の数字を保持
original_numbers = numbered_walls.copy()

lights = np.zeros_like(grid, dtype=int)  # 2:電球

# === 光の計算 ===
def compute_shine(lights):
    shines = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if lights[r, c] == 2:
                shines[r, c] = True
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    while 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == -1 or grid[rr, cc] == 10:
                            break
                        shines[rr, cc] = True
                        rr += dr
                        cc += dc
    return shines

# === 数字マスの残り電球数計算 ===
def remaining_number(lights):
    remain = {}
    for (r,c), v in original_numbers.items():
        cnt = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < rows and 0 <= cc < cols and lights[rr, cc] == 2:
                cnt += 1
        remain[(r,c)] = max(v - cnt, 0)
    return remain

# === 描画関数 ===
def draw_board(ax, lights=None, shines=None, message=None):
    ax.clear()
    ax.set_xticks(np.arange(cols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(rows+1)-0.5, minor=True)
    ax.grid(which="minor", color="k", linewidth=1)
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    remain = remaining_number(lights) if lights is not None else {}

    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            if cell == -1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black"))
            elif cell == 10:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black"))
                val = remain.get((r,c), original_numbers.get((r,c), 0))
                text = "😄" if val == 0 else str(val)
                ax.text(c, r, text, color="white", ha="center", va="center", fontsize=14, weight="bold")
            elif shines is not None and shines[r, c]:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="yellow", alpha=0.3))

            if lights is not None and lights[r, c] == 2:
                ax.plot(c, r, "o", color="orange", markersize=18)

    if message:
        ax.text(cols/2 - 0.5, rows + 0.3, message, color="green", fontsize=18, ha="center")

# === クリア判定 ===
def check_clear(lights):
    shines = compute_shine(lights)
    for r in range(rows):
        for c in range(cols):
            if grid[r,c] == 0 and not shines[r,c]:
                return False, shines
    for r in range(rows):
        for c in range(cols):
            if lights[r,c] == 2:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    while 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == -1 or grid[rr, cc] == 10:
                            break
                        if lights[rr, cc] == 2:
                            return False, shines
                        rr += dr
                        cc += dc
    remain = remaining_number(lights)
    if any(v > 0 for v in remain.values()):
        return False, shines
    return True, shines

# === イベント処理 ===
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)  # ボタン用の余白確保

def onclick(event):
    global lights
    if event.inaxes != ax: return
    c, r = int(round(event.xdata)), int(round(event.ydata))
    if 0 <= r < rows and 0 <= c < cols:
        if grid[r,c] == 0:
            if lights[r,c] == 2:
                lights[r,c] = 0  # 既に置いてあれば消す
            else:
                # 周囲の数字付き黒マスが0なら置けない
                blocked = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < rows and 0 <= cc < cols and grid[rr, cc] == 10:
                        if remaining_number(lights).get((rr,cc), original_numbers[(rr,cc)]) == 0:
                            blocked = True
                if not blocked:
                    lights[r,c] = 2  # 置く

    is_clear, shines = check_clear(lights)
    msg = "🎉 Clear!" if is_clear else None
    draw_board(ax, lights, shines, msg)
    fig.canvas.draw_idle()


# === リセットボタン ===
def reset(event):
    global lights
    lights.fill(0)
    draw_board(ax)
    fig.canvas.draw_idle()

ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
button = Button(ax_button, 'Reset')
button.on_clicked(reset)

draw_board(ax)
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
