import pygame
from environment import ShoverWorldEnv, BOX, EMPTY, LAVA, BARRIER, OBSTACLE
from utils import encode_action

CELL = 48
HUD_HEIGHT = 120


def run_gui(env: ShoverWorldEnv):
    pygame.init()
    rows, cols = env.n_rows, env.n_cols
    screen = pygame.display.set_mode((cols * CELL, rows * CELL + HUD_HEIGHT))
    pygame.display.set_caption("ShoverWorld")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    big_font = pygame.font.SysFont(None, 46, bold=True)

    selected_box = None
    last_info = {}
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                c = mx // CELL
                r = my // CELL
                if 0 <= r < env.n_rows and 0 <= c < env.n_cols and env.grid[r, c] == BOX:
                    selected_box = (r, c)
                    env.previous_selected_position = (r, c)
                else:
                    selected_box = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    selected_box = None

                elif event.key == pygame.K_b:
                    a = encode_action(0, 0, 5, env.n_rows, env.n_cols)
                    obs, rew, done, info = env.step(a)
                    last_info = info

                elif event.key == pygame.K_h:
                    a = encode_action(0, 0, 6, env.n_rows, env.n_cols)
                    obs, rew, done, info = env.step(a)
                    last_info = info

                elif selected_box is not None and event.key in (
                    pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                    pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d
                ):
                    r, c = selected_box
                    if event.key in (pygame.K_UP, pygame.K_w):
                        act = 1
                        dr, dc = -1, 0
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        act = 2
                        dr, dc = 0, 1
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        act = 3
                        dr, dc = 1, 0
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        act = 4
                        dr, dc = 0, -1
                    else:
                        continue

                    obs, rew, done, info = env.step(((r, c), act))
                    last_info = info
                    nr, nc = r + dr, c + dc
                    if info.get("last_action_valid?", False) and 0 <= nr < env.n_rows and 0 <= nc < env.n_cols and env.grid[nr, nc] == BOX:
                        selected_box = (nr, nc)
                    else:
                        if 0 <= r < env.n_rows and 0 <= c < env.n_cols and env.grid[r, c] == BOX:
                            selected_box = (r, c)
                        else:
                            selected_box = None

        # draw grid
        screen.fill((30, 30, 30))
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                val = int(env.grid[r, c])
                rect = pygame.Rect(c * CELL, r * CELL, CELL - 1, CELL - 1)
                if val == EMPTY:
                    color = (240, 240, 240)
                elif val == BOX:
                    color = (255, 105, 180)
                elif val == BARRIER:
                    color = (110, 110, 110)
                elif val == LAVA:
                    color = (200, 60, 60)
                elif val == OBSTACLE:
                    color = (120, 120, 120)
                else:
                    color = (70, 200, 70)

                pygame.draw.rect(screen, color, rect)
                if selected_box == (r, c):
                    pygame.draw.rect(screen, (50, 200, 250), rect, 3)

        hud_y = env.n_rows * CELL
        info_text = f"Stage: {env.level}    t={env.timestep}    stamina={env.stamina:.1f}    boxes={int((env.grid==BOX).sum())}    score={env.score:.1f}"
        perf_info = ", ".join([f"{p['n']}@{p['top_left']}" for p in env.perf_squares])
        screen.blit(font.render(info_text, True, (255, 255, 255)), (8, hud_y + 6))
        screen.blit(font.render(f"perf_squares: {perf_info}", True, (255, 255, 255)), (8, hud_y + 28))
        screen.blit(font.render("Click box -> Arrow/WASD | B=BarrierMaker | H=Hellify | R=reset | Q=quit", True, (200, 200, 200)), (8, hud_y + 54))

        # win overlay
        if env.last_won_info is not None and env.win_flash_timer > 0:
            msg = env.last_won_info.get("message", "you win :)")
            overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            screen.blit(overlay, (0, 0))
            txt_surf = big_font.render(msg, True, (255, 255, 255))
            tw, th = txt_surf.get_size()
            screen.blit(txt_surf, ((screen.get_width() - tw) // 2, (screen.get_height() - th) // 2 - 10))
            env.win_flash_timer = max(0, env.win_flash_timer - 1)
            if env.win_flash_timer == 0:
                env.last_won_info = None

        # lost overlay
        if env.last_lost_info is not None and env.lost_flash_timer > 0:
            msg = env.last_lost_info.get("message", "you lose :(")
            overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            txt_surf = big_font.render(msg, True, (255, 70, 70))
            tw, th = txt_surf.get_size()
            screen.blit(txt_surf, ((screen.get_width() - tw) // 2, (screen.get_height() - th) // 2))
            env.lost_flash_timer = max(0, env.lost_flash_timer - 1)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
