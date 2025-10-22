# Create an improved version that (1) constrains targets/objects inside the arm’s reachable
# workspace, (2) moves the bin closer by default, and (3) draws the reachable workspace ring.
# robot_ai_pygame_reachsafe.py
# Robot + AI demo with reach-safe spawning, goal clamping, and closer bin.
# Author: Z.Ding
# Date: 2025-10-21

import sys, math, random, re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame

Vec2 = Tuple[float, float]

def clamp(x, lo, hi): return max(lo, min(hi, x))
def dist(a: Vec2, b: Vec2) -> float: return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
def lerp(a, b, t): return a + (b - a) * t

@dataclass
class Arm2D:
    base: Vec2
    L1: float = 180.0
    L2: float = 140.0
    q1: float = math.radians(5)
    q2: float = math.radians(0)
    dq1: float = 0.0
    dq2: float = 0.0
    q1_lim: Tuple[float, float] = (math.radians(-135), math.radians(135))
    q2_lim: Tuple[float, float] = (math.radians(-150), math.radians(150))
    max_speed: float = math.radians(180)
    damping: float = 0.1

    def fk(self):
        x0, y0 = self.base
        x1 = x0 + self.L1 * math.cos(self.q1)
        y1 = y0 + self.L1 * math.sin(self.q1)
        x2 = x1 + self.L2 * math.cos(self.q1 + self.q2)
        y2 = y1 + self.L2 * math.sin(self.q1 + self.q2)
        return (self.base, (x1, y1), (x2, y2))

    def ee(self): return self.fk()[2]

    def step_jacobian_transpose(self, target: Vec2, dt: float, kp: float = 0.005, ka: float = 6.0):
        p0, p1, p2 = self.fk()
        ex = target[0] - p2[0]; ey = target[1] - p2[1]

        s1 = math.sin(self.q1); c1 = math.cos(self.q1)
        s12 = math.sin(self.q1 + self.q2); c12 = math.cos(self.q1 + self.q2)
        J = [
            [-self.L1*s1 - self.L2*s12, -self.L2*s12],
            [ self.L1*c1 + self.L2*c12,  self.L2*c12],
        ]
        e = (ex, ey)
        dq1_cmd = J[0][0]*e[0] + J[1][0]*e[1]
        dq2_cmd = J[0][1]*e[0] + J[1][1]*e[1]

        self.dq1 = lerp(self.dq1, kp*dq1_cmd, ka*dt)
        self.dq2 = lerp(self.dq2, kp*dq2_cmd, ka*dt)

        v1 = clamp(self.dq1, -self.max_speed, self.max_speed)
        v2 = clamp(self.dq2, -self.max_speed, self.max_speed)

        self.q1 += v1 * dt; self.q2 += v2 * dt
        self.q1 = clamp(self.q1, *self.q1_lim); self.q2 = clamp(self.q2, *self.q2_lim)
        return (ex*ex + ey*ey) ** 0.5

COLORS = {
    "red": (220, 60, 60),
    "green": (60, 200, 100),
    "blue": (70, 120, 230),
    "yellow": (240, 210, 60),
    "white": (240, 240, 240),
    "black": (15, 15, 18),
    "gray": (120, 130, 140),
    "cyan": (50, 210, 210),
    "purple": (160, 90, 220),
}

@dataclass
class Item:
    pos: Vec2
    color_name: str
    radius: int = 12
    held: bool = False

@dataclass
class Bin:
    rect: pygame.Rect
    visible: bool = True

@dataclass
class AIController:
    current_goal: Optional[Vec2] = None
    mode: str = "idle"  # idle | pick | place | move
    target_color: Optional[str] = None
    held_item: Optional[Item] = None

    def parse(self, text: str) -> str:
        s = text.strip().lower()
        m = re.search(r"move\s*to\s*\(([-\d\.]+)\s*,\s*([-\d\.]+)\)", s)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            self.current_goal = (x, y); self.mode = "move"
            return f"Okay, moving to ({x:.0f},{y:.0f})."
        if "toggle bin" in s: return "TOGGLE_BIN"
        if "spawn object" in s or "spawn objects" in s or "spawn" in s: return "SPAWN_OBJECTS"
        if "reset" in s: return "RESET"
        if "place" in s: self.mode = "place"; return "Placing the held item into the bin."
        if "pick red" in s or "grab red" in s: self.mode = "pick"; self.target_color="red"; return "Picking a red object."
        if "pick green" in s or "grab green" in s: self.mode = "pick"; self.target_color="green"; return "Picking a green object."
        if "pick blue" in s or "grab blue" in s: self.mode = "pick"; self.target_color="blue"; return "Picking a blue object."
        return "Sorry, try: 'pick red', 'place', or 'move to (600,350)'."

    def plan(self, arm: Arm2D, items: List[Item], bin_obj: Optional[Bin]) -> Optional[Vec2]:
        if self.mode == "move" and self.current_goal:
            return self.current_goal
        if self.mode == "pick":
            if self.held_item:
                if bin_obj and bin_obj.visible:
                    cx = bin_obj.rect.centerx; cy = bin_obj.rect.top - 10
                    self.current_goal = (cx, cy); return self.current_goal
                self.mode = "idle"; return None
            cands = [it for it in items if not it.held and it.color_name == self.target_color]
            if not cands: self.mode = "idle"; return None
            ee = arm.ee()
            it = min(cands, key=lambda o: dist(ee, o.pos))
            self.current_goal = (it.pos[0], it.pos[1]-5); return self.current_goal
        if self.mode == "place":
            if self.held_item and bin_obj and bin_obj.visible:
                cx = bin_obj.rect.centerx; cy = bin_obj.rect.top - 10
                self.current_goal = (cx, cy); return self.current_goal
            self.mode = "idle"; return None
        return None

class RobotAIDemo:
    def __init__(self, w=900, h=600):
        pygame.init()
        pygame.display.set_caption("Robot + AI Demo")
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)

        # Arm/base
        self.arm = Arm2D(base=(140, h//2 + 120))
        self.r_min = abs(self.arm.L1 - self.arm.L2) + 8.0
        self.r_max = (self.arm.L1 + self.arm.L2) - 14.0

        # Items & bin
        self.items: List[Item] = []
        # Move bin CLOSER by default: place it near the right edge of reachable ring
        bx, by = self.project_into_workspace((self.arm.base[0] + self.r_max - 20, self.arm.base[1] - 10))
        self.bin = Bin(rect=pygame.Rect(int(bx)-60, int(by)+10, 120, 90), visible=True)

        self.ai = AIController()
        self.ai_enabled = True
        self.help_visible = True
        self.goal: Optional[Vec2] = None
        self.hold_threshold = 16.0
        self.spawn_random_objects(6)

    # ---------- workspace helpers ----------
    def project_into_workspace(self, p: Vec2) -> Vec2:
        """Clamp a point into the annular reachable set [r_min, r_max] around arm.base."""
        bx, by = self.arm.base
        dx, dy = p[0]-bx, p[1]-by
        r = (dx*dx + dy*dy) ** 0.5
        if r < 1e-6:
            return (bx + self.r_min, by)  # arbitrary direction
        r_clamped = clamp(r, self.r_min, self.r_max)
        scale = r_clamped / r
        # keep inside screen bounds slightly
        x = bx + dx * scale
        y = by + dy * scale
        W, H = self.screen.get_size()
        x = clamp(x, 10, W-10); y = clamp(y, 10, H-10)
        return (x, y)

    def spawn_random_objects(self, n=5):
        self.items.clear()
        colors = ["red", "green", "blue"]
        for _ in range(n):
            # sample a random angle and radius inside reachable ring
            ang = random.uniform(-2.3, -0.2) if random.random() < 0.15 else random.uniform(-math.pi*0.15, -math.pi*0.85)
            r = random.uniform(self.r_min+10, self.r_max-30)
            x = self.arm.base[0] + r*math.cos(ang)
            y = self.arm.base[1] + r*math.sin(ang)
            x, y = self.project_into_workspace((x, y))
            c = random.choice(colors)
            self.items.append(Item(pos=(x, y), color_name=c))

    def snap_goal_to_color(self, color_name: str):
        ee = self.arm.ee()
        cands = [it for it in self.items if not it.held and it.color_name == color_name]
        if not cands: return
        it = min(cands, key=lambda o: dist(ee, o.pos))
        self.goal = self.project_into_workspace(it.pos)

    def text_input_dialog(self, prompt="Enter prompt:"):
        W, H = self.screen.get_size(); txt = ""; done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN: done = True
                    elif event.key == pygame.K_BACKSPACE: txt = txt[:-1]
                    elif event.key == pygame.K_ESCAPE: return None
                    else:
                        if event.unicode and len(txt) < 80: txt += event.unicode
            overlay = pygame.Surface((W, 80), pygame.SRCALPHA); overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0, H//2 - 40))
            s1 = self.big.render(prompt, True, (240,240,240))
            s2 = self.big.render(txt + "|", True, (240,210,60))
            self.screen.blit(s1, (20, H//2 - 30)); self.screen.blit(s2, (20, H//2 + 2))
            pygame.display.flip(); self.clock.tick(60)
        return txt

    def reset(self):
        self.arm.q1 = math.radians(5); self.arm.q2 = math.radians(0)
        self.arm.dq1 = self.arm.dq2 = 0.0
        self.ai = AIController(); self.goal = None
        self.spawn_random_objects(6)
        # move bin again closer (in case window changed)
        bx, by = self.project_into_workspace((self.arm.base[0] + self.r_max - 20, self.arm.base[1] - 10))
        self.bin.rect.update(int(bx)-60, int(by)+10, 120, 90)

    # ---------- drawing ----------
    def draw_workspace(self):
        # Visualize reachable ring [r_min, r_max]
        surf = self.screen
        bx, by = self.arm.base
        pygame.draw.circle(surf, (60, 60, 70), (int(bx), int(by)), int(self.r_max), 1)
        pygame.draw.circle(surf, (60, 60, 70), (int(bx), int(by)), int(self.r_min), 1)

    def draw_arm(self):
        p0, p1, p2 = self.arm.fk()
        pygame.draw.circle(self.screen, (120,130,140), (int(p0[0]), int(p0[1])), 10)
        pygame.draw.line(self.screen, (240,240,240), p0, p1, 5)
        pygame.draw.circle(self.screen, (120,130,140), (int(p1[0]), int(p1[1])), 8)
        pygame.draw.line(self.screen, (240,240,240), p1, p2, 5)
        pygame.draw.circle(self.screen, (240,210,60), (int(p2[0]), int(p2[1])), 6)

    def draw_items(self):
        for it in self.items:
            if it.held: continue
            color = {"red":(220,60,60),"green":(60,200,100),"blue":(70,120,230)}[it.color_name]
            pygame.draw.circle(self.screen, color, (int(it.pos[0]), int(it.pos[1])), it.radius)

    def draw_bin(self):
        if not self.bin.visible: return
        pygame.draw.rect(self.screen, (60,60,60), self.bin.rect, border_radius=8)
        lip = pygame.Rect(self.bin.rect.x-8, self.bin.rect.y-8, self.bin.rect.w+16, 10)
        pygame.draw.rect(self.screen, (110,110,110), lip, border_radius=6)
        label = self.font.render("BIN", True, (240,240,240))
        self.screen.blit(label, (self.bin.rect.centerx-14, self.bin.rect.y-26))

    def draw_goal(self):
        if not self.goal: return
        pygame.draw.circle(self.screen, (50,210,210), (int(self.goal[0]), int(self.goal[1])), 6, 2)

    def draw_hud(self, err_px: float):
        ee = self.arm.ee()
        lines = [
            f"AI: {'ON' if self.ai_enabled else 'OFF'}  |  Mode: {self.ai.mode}",
            f"q1={math.degrees(self.arm.q1):5.1f}°, q2={math.degrees(self.arm.q2):5.1f}°   EE=({ee[0]:.0f},{ee[1]:.0f})",
            f"Goal={'None' if not self.goal else f'({self.goal[0]:.0f},{self.goal[1]:.0f})'}   Error={err_px:5.1f}px",
            "Keys: [A]I, [P]rompt, [T] spawn, [B] bin, [R]eset, [1/2/3] color-goal, click=manual, [H] help",
        ]
        for i, L in enumerate(lines):
            s = self.font.render(L, True, (240,240,240)); self.screen.blit(s, (10, 8 + i*18))
        help_lines = [
            "Workspace ring shows reach limits. All targets are auto-clamped inside reach.",
            "PROMPTS: 'pick red/green/blue', 'place', 'move to (x,y)', 'toggle bin', 'spawn objects', 'reset'",
        ]
        for j, L in enumerate(help_lines):
            s = self.font.render(L, True, (120,130,140))
            self.screen.blit(s, (10, 8 + (len(lines)+j)*18))

    # ---------- interaction ----------
    def try_pick_or_place(self):
        ee = self.arm.ee()
        if self.ai.held_item and self.bin.visible and self.bin.rect.collidepoint(ee[0], ee[1]):
            self.ai.held_item.held = False
            rx = random.randint(self.bin.rect.x+12, self.bin.rect.right-12)
            ry = random.randint(self.bin.rect.y+20, self.bin.rect.bottom-12)
            self.ai.held_item.pos = (rx, ry)
            self.ai.held_item = None; self.ai.mode = "idle"
            return "Placed."
        if not self.ai.held_item:
            for it in self.items:
                if not it.held and dist(ee, it.pos) <= 16.0:
                    it.held = True; self.ai.held_item = it
                    return f"Picked {it.color_name}."
        return None

    def update_held_item_pose(self):
        if self.ai.held_item:
            self.ai.held_item.pos = self.arm.ee()

    def mainloop(self):
        running = True; err_px = 0.0; last_info_msg = ""
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_goal = pygame.mouse.get_pos()
                    self.goal = self.project_into_workspace(mouse_goal)
                    self.ai.mode = "idle"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_a: self.ai_enabled = not self.ai_enabled
                    elif event.key == pygame.K_b: self.bin.visible = not self.bin.visible
                    elif event.key == pygame.K_t: self.spawn_random_objects(6)
                    elif event.key == pygame.K_r: self.reset()
                    elif event.key == pygame.K_h: self.help_visible = not self.help_visible
                    elif event.key == pygame.K_p:
                        user_txt = self.text_input_dialog("Prompt:")
                        if user_txt is not None:
                            action = self.ai.parse(user_txt)
                            if action == "TOGGLE_BIN": self.bin.visible = not self.bin.visible; last_info_msg = "Toggled bin."
                            elif action == "SPAWN_OBJECTS": self.spawn_random_objects(6); last_info_msg = "Spawned objects."
                            elif action == "RESET": self.reset(); last_info_msg = "Scene reset."
                            else: last_info_msg = action
                    elif event.key == pygame.K_1: self.snap_goal_to_color("red"); self.ai.mode = "idle"
                    elif event.key == pygame.K_2: self.snap_goal_to_color("green"); self.ai.mode = "idle"
                    elif event.key == pygame.K_3: self.snap_goal_to_color("blue"); self.ai.mode = "idle"

            if self.ai_enabled:
                ag = self.ai.plan(self.arm, self.items, self.bin if self.bin.visible else None)
                if ag is not None:
                    self.goal = self.project_into_workspace(ag)

            if self.goal is not None:
                err_px = self.arm.step_jacobian_transpose(self.goal, dt, kp=0.005, ka=6.0)
            else:
                err_px = 0.0

            self.update_held_item_pose()
            msg = self.try_pick_or_place()
            if msg: last_info_msg = msg

            self.screen.fill((28,28,32))
            self.draw_workspace()
            self.draw_bin(); self.draw_items(); self.draw_arm(); self.draw_goal(); self.draw_hud(err_px)
            if last_info_msg:
                s = self.font.render(last_info_msg, True, (160,90,220))
                self.screen.blit(s, (10, self.screen.get_height()-26))
            pygame.display.flip()
        pygame.quit()

def main():
    demo = RobotAIDemo()
    demo.mainloop()

if __name__ == "__main__":
    main()

