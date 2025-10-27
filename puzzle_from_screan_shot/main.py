#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage examples:
  python main.py input.png --rows 5 --cols 5 > puzzle.json
  python main.py input.png --size 5x5 > puzzle.json
  python main.py input.png --size-file meta.json > puzzle.json
"""
import argparse
import json
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np

@dataclass
class Params:
    black_v_thresh: float = 40.0    
    dash_white_thresh: float = 160.0 
    dash_area_frac_max: float = 0.10 
    dash_height_frac_max: float = 0.35
    dash_width_height_ratio_min: float = 2.0
    dash_width_frac_min: float = 0.10

def order_poly_corners(pts4: np.ndarray) -> np.ndarray:
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def find_board_quad(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h*w)*0.1:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    if best is None:
        if not contours:
            raise RuntimeError("Board not found: no contours")
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        best = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(best) != 4:
            x,y,w2,h2 = cv2.boundingRect(cnt)
            best = np.array([[[x,y]], [[x+w2,y]], [[x+w2,y+h2]], [[x,y+h2]]], dtype=np.int32)

    quad = best.reshape(-1, 2).astype(np.float32)
    return order_poly_corners(quad)

def rectify_board(img_bgr: np.ndarray, quad: np.ndarray, out_w: int=1000, out_h: int=1000) -> np.ndarray:
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return warped

def classify_black_or_open(cell_bgr: np.ndarray, params: Params) -> str:
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[...,2].astype(np.float32)
    return "black" if float(v.mean()) < params.black_v_thresh else "open"

def has_white_number_roi(gray_roi: np.ndarray, white_thresh: int) -> bool:
    _, bw = cv2.threshold(gray_roi, int(white_thresh), 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    area = bw.sum() / 255.0
    h, w = bw.shape
    frac = area / float(h*w)
    return frac >= 0.01

def count_horizontal_dashes(cell_bgr: np.ndarray, params: Params) -> int:
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pad_y = int(h*0.12); pad_x = int(w*0.12)
    if h > 2*pad_y and w > 2*pad_x:
        roi = gray[pad_y:h-pad_y, pad_x:w-pad_x]
    else:
        roi = gray

    _, bw = cv2.threshold(roi, int(params.dash_white_thresh), 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    Hc, Wc = bw.shape
    area_cap = params.dash_area_frac_max * (Hc*Wc)
    good = 0
    for i in range(1, num):
        x,y,wc,hc,area = stats[i]
        if area <= area_cap and wc > hc*params.dash_width_height_ratio_min and hc <= Hc*params.dash_height_frac_max and wc >= Wc*params.dash_width_frac_min:
            good += 1
    return max(0, min(4, int(good)))

def extract_grid(img_bgr: np.ndarray, rows: int, cols: int, params: Params):
    H, W = img_bgr.shape[:2]
    cell_w = W / cols
    cell_h = H / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            xs = int(round(c*cell_w)); xe = int(round((c+1)*cell_w))
            ys = int(round(r*cell_h)); ye = int(round((r+1)*cell_h))
            cell = img_bgr[ys:ye, xs:xe]
            kind = classify_black_or_open(cell, params)
            if kind == "open":
                out.append(".")
            else:
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                ch, cw = gray.shape
                py = int(ch*0.12); px = int(cw*0.12)
                roi = gray[py:ch-py, px:cw-px] if (ch>2*py and cw>2*px) else gray
                if has_white_number_roi(roi, params.dash_white_thresh):
                    d = count_horizontal_dashes(cell, params)
                    out.append(f"#{d}")
                else:
                    out.append("#")
    return out

def parse_size_args(rows: Optional[int], cols: Optional[int], size: Optional[str], size_file: Optional[str]) -> Tuple[int,int]:
    if rows is not None and cols is not None:
        return int(rows), int(cols)
    if size:
        try:
            r,c = size.lower().replace("×","x").split("x")
            return int(r), int(c)
        except Exception as e:
            raise SystemExit(f"Invalid --size format. Use like --size 5x5. error={e}")
    if size_file:
        with open(size_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        r = meta.get("rows", meta.get("height"))
        c = meta.get("cols", meta.get("width"))
        if r is None or c is None:
            raise SystemExit("size-file must contain rows/cols or height/width")
        return int(r), int(c)
    raise SystemExit("Please provide grid size by --rows/--cols or --size or --size-file.")

def main():
    ap = argparse.ArgumentParser(description="Akari/Light-Up screenshot to JSON grid")
    ap.add_argument("image", help="input screenshot path")
    ap.add_argument("--rows", type=int, help="number of rows in the grid")
    ap.add_argument("--cols", type=int, help="number of cols in the grid")
    ap.add_argument("--size", type=str, help="grid size like '5x5' or '5×5'")
    ap.add_argument("--size-file", type=str, help="JSON file containing {'rows':R,'cols':C}")
    ap.add_argument("--black-v-thresh", type=float, default=Params.black_v_thresh, help="V mean threshold for black cells")
    ap.add_argument("--dash-white-thresh", type=float, default=Params.dash_white_thresh, help="binary threshold for counting dashes")
    ap.add_argument("--debug-rectified", help="optional path to save rectified board preview")
    ap.add_argument("--flat", action="store_true", help="output grid as a flat list instead of 2D rows")
    ap.add_argument("--compact-rows", action="store_true", help="print each row on a single line in JSON output")
    ap.add_argument("--row-join", type=str, help="join each row into a single string using this delimiter (e.g., '', ',', or ' ')")
    ap.add_argument("--single-line-json", action="store_true", help="print the whole JSON in one line")
    ap.add_argument("--flat-wrap", action="store_true", help="when using --flat, wrap the flat list with line breaks every WIDTH items")
    ap.add_argument("--emit-grid-body", action="store_true", help="print only the inner contents of grid (omit the enclosing [ and ])")
    ap.add_argument("--rows-no-brackets", action="store_true", help="omit [ ] around each row when emitting 2D rows")
    args = ap.parse_args()

    params = Params(
        black_v_thresh=args.black_v_thresh,
        dash_white_thresh=args.dash_white_thresh
    )

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    quad = find_board_quad(img)
    rows, cols = parse_size_args(args.rows, args.cols, args.size, args.size_file)

    out_size = max(rows, cols) * 120
    warped = rectify_board(img, quad, out_w=out_size, out_h=out_size)

    if args.debug_rectified:
        cv2.imwrite(args.debug_rectified, warped)

    grid = extract_grid(warped, rows, cols, params)
    grid2d = [grid[r*cols:(r+1)*cols] for r in range(rows)] if not args.flat else grid
    if not args.flat and args.row_join is not None:
        joiner = args.row_join
        grid_out = [joiner.join(row) for row in grid2d]
    else:
        grid_out = grid2d
    puzzle = {
        "width": cols,
        "height": rows,
        "grid": grid_out
    }
    if args.emit_grid_body:
        def esc(x):
            return json.dumps(x, ensure_ascii=False)
        if args.flat:
            if args.flat_wrap:
                chunks = [grid_out[i:i+cols] for i in range(0, len(grid_out), cols)]
                for i,chunk in enumerate(chunks):
                    line = ", ".join(esc(v) for v in chunk)
                    suffix = "," if i < len(chunks)-1 else ""
                    print("    " + line + suffix)
            else:
                print(", ".join(esc(v) for v in grid_out))
        else:
            if args.row_join is not None:
                for i,row in enumerate(grid2d):
                    line = esc((args.row_join).join(row))
                    suffix = "," if i < len(grid2d)-1 else ""
                    print("    " + line + suffix)
            else:
                for i,row in enumerate(grid2d):
                    if args.rows_no_brackets:
                        line = ", ".join(esc(v) for v in row)
                    else:
                        line = "[" + ", ".join(esc(v) for v in row) + "]"
                    suffix = "," if i < len(grid2d)-1 else ""
                    print("    " + line + suffix)
    elif args.flat and args.flat_wrap and isinstance(grid_out, list):
        def esc(x):
            return json.dumps(x, ensure_ascii=False)
        chunks = [grid_out[i:i+cols] for i in range(0, len(grid_out), cols)]
        lines = ["    " + ", ".join(esc(v) for v in chunk) for chunk in chunks]
        grid_block = ",\n".join(lines)
        print("{\n  \"width\": ", cols, ",\n  \"height\": ", rows, ",\n  \"grid\": [\n" + grid_block + "\n  ]\n}", sep="")
    elif args.single_line_json:
        print(json.dumps(puzzle, ensure_ascii=False, separators=(",", ":")))
    elif args.compact_rows and not args.flat and args.row_join is None:
        def esc(x):
            return json.dumps(x, ensure_ascii=False)
        row_lines = ["  "*2 + "[" + ", ".join(esc(c) for c in row) + "]" for row in grid2d]
        body = ",\n".join(row_lines)
        print("{\n  \"width\": ", cols, ",\n  \"height\": ", rows, ",\n  \"grid\": [\n" + body + "\n  ]\n}", sep="")
    else:
        print(json.dumps(puzzle, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
