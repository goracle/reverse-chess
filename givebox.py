#!/usr/bin/python3
"""reverse chess' gift module"""
import sys
import re
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from PIL import ImageFont
from PIL import ImageDraw
import PIL
import pygame  # <--- ADD THIS LINE
# module
from chessassets import IMAGES, PTYPES, offboard_count

def number_image(num):
    """create number image"""
    img = Image.new('RGB', (20,20), (250,250,250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("OpenSans-Regular.ttf", 18)
    #draw.text((150, -30),str(num),(0,0,0),font=font)
    draw.text((0, 0),str(num),(0,0,0),font=font, align='center')
    #img.save('digit_number_img_'+str(i)+'.jpg')
    return img

global CLICKED
global COMMANDS
global ROOT
ROOT = None
CLICKED = {}
COMMANDS = {}
for ptype2 in ['p', 'r', 'n', 'q', 'k', 'b']:
    CLICKED[ptype2] = 0
    def click(gset=None, ptype=ptype2):
        global CLICKED
        global COMMANDS
        global ROOT
        CLICKED[ptype] += 1
        if gset is not None:
            CLICKED[ptype] = gset
        else:
            if ROOT is not None:
                on_close()
        return CLICKED[ptype]
    COMMANDS[ptype2] = click

global ICONS
ICONS = [] 

def on_close():
    """close the button window"""
    global ROOT
    ROOT.quit()
    ROOT.destroy()
    ROOT = None

def on_close():
    """close the button window"""
    global ROOT
    if ROOT is not None:
        ROOT.destroy()  # Just destroy, don't call quit()
        ROOT = None


def reset_clicked(clicked):
    """reset the click count"""
    global CLICKED
    global COMMANDS
    global ROOT
    for ptype in CLICKED:
        CLICKED[ptype] = 0
        assert not COMMANDS[ptype](0)
    ROOT = None


def givebox(pieces, ignore_count=False):
    global CLICKED
    global COMMANDS
    global ROOT
    global ICONS

    # Use existing root or create if none exists
    if tk._default_root is None:
        root = tk.Tk()
    else:
        root = tk.Toplevel()  # Create a child window instead
    
    root.geometry('500x500')


    root.resizable(True, True)
    root.title('Which piece to give back?'\
               if not ignore_count else\
               'promote pawn to which piece?')
    root.protocol('WM_DELETE_WINDOW',on_close)
    canvas = tk.Canvas(root, width = 500, height = 500)

    clicked = reset_clicked(CLICKED)
    ROOT = root
    assert ROOT is not None, ROOT

    button_made = {}
    if not ignore_count:
        counts = offboard_count(pieces)

    pady = 15
    padx = 20
    button_count = 0
    for piece in pieces:
        team = piece.team
        ptype = piece.ptype
        if ptype in ('p', 'k') and ignore_count: # ignore_count is used for promotions
            continue
        if not ignore_count:
            count = counts[ptype]
            if not count:
                continue
        if piece.ptype not in button_made:
            button_made[ptype] = True
            command = COMMANDS[ptype]

            image_str = IMAGES[(team, ptype, False)]
            image=Image.open(image_str)
            if not ignore_count:
                add_count_to_image(image, 'P', image_str)
                image = add_count_to_image(image, count, image_str)
            icon = PIL.ImageTk.PhotoImage(image, master=canvas)
            ICONS.append(icon)
            button = ttk.Button(root, image=icon, command=command,)
            button.grid(row=button_count//2, column=button_count%2, ipady=30, ipadx=30)
            button_count += 1

    root.mainloop()
    ret = None
    for icon in ICONS:
        pass
    for ptype in CLICKED:
        if CLICKED[ptype]:
            if ignore_count:
                return ptype # just give the ptype for this mode
            for piece in pieces:
                if piece.ptype == ptype and not piece.ontheboard:
                    ret = piece
                    break
            assert ret is not None, ret
            assert CLICKED[ptype] == 1, (ptype, CLICKED[ptype])
            break
        else:
            assert not CLICKED[ptype], CLICKED
    if ignore_count:
        print("givebox, ignore count:")
        piece.ident()
    return ret

def add_count_to_image(image, count, image_str):
    """add a count of off-board pieces of a ptype
    to image of that piece type"""
    im1 = number_image(count) # number of pieces not on the board of that type/team
    if count == 'P':
        image.paste(im1, (2,2))
        image_str = re.sub(".png", "_p.png", image_str)
        image.save(image_str)
    else:
        image.paste(im1, (0,0))
    return image

    #showinfo(
    #    title='Information',
    #    message='Download button clicked! '+str(download_clicked.a))


if __name__ == '__main__':
    pass
    #givebox()


def givebox(pieces, ignore_count=False):
    global CLICKED
    global COMMANDS
    global ROOT
    global ICONS
    
    # Use existing root or create if none exists
    if tk._default_root is None:
        root = tk.Tk()
        root.withdraw()  # Hide the root
        dialog = tk.Toplevel(root)
    else:
        dialog = tk.Toplevel()  # Create a child window instead
    
    dialog.geometry('500x500')
    dialog.resizable(True, True)
    dialog.title('Which piece to give back?'\
               if not ignore_count else\
               'promote pawn to which piece?')
    dialog.protocol('WM_DELETE_WINDOW', on_close)
    canvas = tk.Canvas(dialog, width = 500, height = 500)
    
    clicked = reset_clicked(CLICKED)
    ROOT = dialog  # Changed from root to dialog
    assert ROOT is not None, ROOT
    
    button_made = {}
    if not ignore_count:
        counts = offboard_count(pieces)
    
    pady = 15
    padx = 20
    button_count = 0
    for piece in pieces:
        team = piece.team
        ptype = piece.ptype
        if ptype in ('p', 'k') and ignore_count:
            continue
        if not ignore_count:
            count = counts[ptype]
            if not count:
                continue
        if piece.ptype not in button_made:
            button_made[ptype] = True
            command = COMMANDS[ptype]
            
            image_str = IMAGES[(team, ptype, False)]
            image=Image.open(image_str)
            if not ignore_count:
                add_count_to_image(image, 'P', image_str)
                image = add_count_to_image(image, count, image_str)
            icon = PIL.ImageTk.PhotoImage(image, master=canvas)
            ICONS.append(icon)
            button = ttk.Button(dialog, image=icon, command=command,)
            button.grid(row=button_count//2, column=button_count%2, ipady=30, ipadx=30)
            button_count += 1
    
    # Use wait_window instead of mainloop
    dialog.wait_window()  # This blocks until dialog is destroyed
    
    ret = None
    for icon in ICONS:
        pass
    for ptype in CLICKED:
        if CLICKED[ptype]:
            if ignore_count:
                return ptype
            for piece in pieces:
                if piece.ptype == ptype and not piece.ontheboard:
                    ret = piece
                    break
            assert ret is not None, ret
            assert CLICKED[ptype] == 1, (ptype, CLICKED[ptype])
            break
        else:
            assert not CLICKED[ptype], CLICKED
    if ignore_count and ret:
        print("givebox, ignore count:")
        piece.ident()
    return ret


def givebox(pieces, ignore_count=False):
    global CLICKED
    global COMMANDS
    global ROOT
    global ICONS
    
    # Create root if it doesn't exist
    if tk._default_root is None:
        hidden_root = tk.Tk()
        hidden_root.withdraw()
    
    dialog = tk.Toplevel()
    dialog.geometry('500x500')
    dialog.resizable(True, True)
    dialog.title('Which piece to give back?'\
               if not ignore_count else\
               'promote pawn to which piece?')
    
    # Don't use protocol - we'll handle it differently
    canvas = tk.Canvas(dialog, width = 500, height = 500)
    
    clicked = reset_clicked(CLICKED)
    ROOT = dialog
    
    button_made = {}
    if not ignore_count:
        counts = offboard_count(pieces)
    
    pady = 15
    padx = 20
    button_count = 0
    for piece in pieces:
        team = piece.team
        ptype = piece.ptype
        if ptype in ('p', 'k') and ignore_count:
            continue
        if not ignore_count:
            count = counts[ptype]
            if not count:
                continue
        if piece.ptype not in button_made:
            button_made[ptype] = True
            command = COMMANDS[ptype]
            
            image_str = IMAGES[(team, ptype, False)]
            image=Image.open(image_str)
            if not ignore_count:
                add_count_to_image(image, 'P', image_str)
                image = add_count_to_image(image, count, image_str)
            icon = PIL.ImageTk.PhotoImage(image, master=canvas)
            ICONS.append(icon)
            button = ttk.Button(dialog, image=icon, command=command,)
            button.grid(row=button_count//2, column=button_count%2, ipady=30, ipadx=30)
            button_count += 1
    
    # Manual event loop - pump both Tk and pygame events

    dialog.update()
    done = False
    while not done:
        
        # 1. Process Pygame events to keep it from freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the main window is closed, exit everything
                pygame.quit()
                sys.exit()
        
        # 2. Process Tkinter events
        try:
            dialog.update_idletasks()
            dialog.update()
        except tk.TclError:
            # Window was closed (e.g., by the 'X' button)
            done = True
            break
        
        # 3. Check if a button was clicked
        for ptype in CLICKED:
            if CLICKED[ptype]:
                done = True
                break
        
        # 4. Small delay to prevent 100% CPU usage
        import time
        time.sleep(0.01)    

    # Clean up
    try:
        dialog.destroy()
    except:
        pass
    
    ROOT = None
    
    # Return the selected piece
    ret = None
    for ptype in CLICKED:
        if CLICKED[ptype]:
            if ignore_count:
                return ptype
            for piece in pieces:
                if piece.ptype == ptype and not piece.ontheboard:
                    ret = piece
                    break
            if ret:
                break
    
    return ret
