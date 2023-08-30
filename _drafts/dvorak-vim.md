---
title: 'From Vim & QWERTY to Neovim & DVORAK'
date_created: '2023-07-25'
date: '2023-07-25'
categories:
  - Programming
excerpt: And never back again.
toc: true
toc_sticky: true

---

I'm a Neovim, DVORAK & split keyboard user.

This post details my transitions to using these tools:

- from Atom to Vim to Neovim,
- from a QWERTY to DVORAK keyboard layout,
- from a traditional to a split keyboard.

# My Journey

I started my programming journey in early 2017 - on Windows laptop using the now deceased Atom editor.  

I didn't know any better!

Programming is both my hobby and profession - I enjoy working on improving my tools.  I like the idea of optimizing my workflows, especially if it improves the health of my aging, tired body.

## Vim

I started to learn Vim in the Christmas holidays of 2018 - I cannot remember why.

The first few days were tough - it took around a week to feel comfortable with the basics of Vim such as `hjkl`, moving between splits or moving to different places a file.

After two weeks I felt as productive as I was in Atom - beyond that my productivity has become more powerful than you could possibly imagine.  Over the years I added colorschemes, plugins, keybinds, macros & abbreviations - [you can find my final `.vimrc` here]().

The initial configuration for Vim can be challenging - I would budget 1-2 days to get a basic setup working.

Alongside Vim I use Tmux and fzf.  Both of these tools are crucial for making Vim as your main text editor a productive experience - without any of the three, my setup would not work.

Tmux is used for terminal multiplexing - allowing the ability to open terminal windows alongside each other.

I use fzf for finding and opening files, both from the shell with `$ vim **<TAB>` and within Vim using `<Space>` to run fzf in the current directory via a keybinding.

## DVORAK

I started my transition to DVORAK in May 2019 - motivated by a desire to improve the health of my hands.

After two and a half years of furious programming, I was suffering with muscular soreness & tiredness in my hands.  My hands felt fatigued - like they were doing too much work.

I was aware that there were alternative keyboard layouts, designed to be kinder to our hands - a day or two of sporadic, random & repetitive searching on Google about the different options (such as Colemak), I decided to give Dvorak a shot.

It took me around 2 weeks to get back to a somewhat reasonable level of productivity, but I was not back to the my previous level of QWERTY.

My typing remained inaccurate for a long time - I only felt like I was back to where I was in August 2021 - making the transition over a year long.

Sometimes my typing is less accurate than it was (particularly as I use a keyboard with blank keycaps) - but it's manageable. 

I don't feel like DVORAK led to a significant productivity improvement. It did help the health of my hands - that benefit was instant.

## Split Keyboard

I started using a split keyboard in July 2021 - motivated by a desire to improve the health of my back.

Previously I had used the Apple Keyboard, then moving to a [Vortex Race 3](https://vortexgear.store/products/race-3-micro-usb), which I still have today.  The split keyboard I use today is the [Ergodox EZ](https://ergodox-ez.com), which I love.

The major benefit of a split keyboard is that your hands rest further apart.  This allows your chest to expand, and reduces the strain on your upper and middle back - in particular reducing pain between the shoulder blades.

It took around 1 week to get back to the same level of productivity as a QWERTY keyboard.  Beyond that I have found some level of productivity increase through the ability to customize the keyboard layout, but it's not a huge increase.

The Ergodox EZ allows customization the keyboard layout using ORYX - [you can find my layout here](https://configure.zsa.io/ergodox-ez/layouts/vJLGQ/latest/0).  Configuration is straight forward.

## Neovim

I started my transition to Neovim in July 2022 - motivated by the Vimscript 9 schism that has divided the Vim community.

Transitioning from Vim to Neovim after 3 years of Vim was essentially instantaneous - the in-editor experience is very similar.

It took around half a day to convert my `.vimrc` to a functional Lua based setup, followed by a week or two of tweaking my config and adding plugins. I was able to bring along all of my Vimscript plugins, which is a huge selling point of Neovim.

My initial Neovim setup used the colorscheme Dracula, bufferline and lualine, Treesitter for syntax highlighting, Null-Ls for linting & formatting & Telescope for searching and the built-in LSP for running language servers.

I did a small amount of format on save in Vim with Black, so Null-Ls was a big improvement.  I also added a snippet engine Luasnip, which has been great.  The Neovim `cmp` completion engine is also a big improvement over what I had in Vim.

In current Neovim setup I use Cokeline, Mason for LSP configuration & efm for linting and formatting - [you can find my Neovim plugins here](https://github.com/ADGEfficiency/dotfiles/blob/master/nvim/lua/adam/plugins.lua).

# Which Would You Recommend?

## Vim

Vim is amazing - I highly recommend it.  

The plugins and customization are fantastic. Vim reduces the amount of typing I do, and reduces mouse movement.  It allows me to never leave the terminal.

I'm a big believer in old tools being good tools (known as the [Lindy effect](https://en.wikipedia.org/wiki/Lindy_effect)) - with Vim I love knowing I'll never use another editor.

Even if you love VS Code, learning a bit of Vim is useful.  It's almost always available on servers, and it's a better editor than other commonly available editors like nano.

Vim keybindings are also everywhere - you can enable in the shell with `$ set -o vi` (instead of the default Emacs bindings), and many programs like IDEs or browsers will have Vim plugins.

## Neovim

Neovim is an improvement over Vim, and has a bubbling, existing ecosystem - it's highly recommended.  If you are a Vim user it's not a big transition - all of your plugins will work as expected.

It's nice to use Lua for configuration - it's more flexible and feels like a more useful, transferrable skill that Vimscript.

I have found the language servers, linting, formatting and completion experience an improvement over Vim.

## DVORAK

I would not recommend the DVORAK switch - while I'm glad I have done it and wouldn't switch back, it takes a long, long time to get used to.

I can still type QWERTY if needed - it's keyboard & context dependent. I can still type QWERTY on my phone without even realizing it's a different layout.

## Split Keyboard

I would recommend a split keyboard - while there is a cost to transitioning, the pain relief I have felt in my back is well worth it.

I have no problem going back to a normal keyboard - unlike DVORAK, using a split keyboard will not impact your ability to use a normal keyboard.

# Thoughts on DVORAK and Vim

I was already a proficient Vim user when I decided to switch to Dvorak.

Foundational to using either layout is remapping `<CAPSLOCK>` to `<ESCAPE>`. In Vim you use the `<ESCAPE>` key to move from insert to normal mode - easy access to the escape key is essential.

## Why DVORAK?

Most computer keyboards are laid out in QWERTY - named for the keys in the first row. 

The big idea in Dvorak is the importance of the middle row (also known as the home row). 

Vim users know the importance of the home row from `hjkl` - the keys used for cursor movement in Vim.  Dvorak puts all the vowels on the home row - the keys you access the most are closest to your fingers.

The other notable feature of Dvorak is the location of the punctuation characters `' , .` - these are located in a prime position.

An interesting thing about learning DVORAK was that the time to learn keys is long tailed. Some (such as `, .` and `aoeu`) come very easily, while others like `r y f g` took a while.

## Losing hjkl

In Vim hjkl are used for cursor movement - they are the keys you use to move your cursor around a file in normal mode.

In DVORAK you lose the position and order of hjkl.  Initially I considered remapping hjkl to the same position as QWERTY, but decided against it.

## Combinations That Work Great

There are some common Vim key combinations that feel great in Dvorak:

- `:` (Vim command mode) is easy access - making `:w`, `:wq` great - you don't need to move either hand
- `.` is easy access,
- `.py` are all next to each other,
- `ls` are right next to each other,
- `gcc`,
- `<C-r>` - no hand movement

`dft`, `dit`
dtv, dfv, di(

## Challenges

The main challenge is anything `g` or `f` related.  In Vim `gf` opens a file under the cursor - as these two keys are next to each other, it's not a smooth as something like `:w`.

# Summary

Here is a summary of the transitions - years of human experience reduced to a Markdown table:

| Transition     | Time to Transition | Config Required | Productivity Increase | Health Improvement | Recommended |
|----------------|-------------------|-----------------|-----------------------|-------------------|-------------|
| Atom to Vim            | 2 weeks           | 1 day           | High                  | None | Yes         |
| Vim to Neovim         | 1/2 day            | 1/2 day         | Moderate | None | Yes         |
| QWERTY to DVORAK         | >1 year           | None       | None                  | Hand fatigue      | No          |
| Split Keyboard | 3 weeks           | 1/2 day         | Moderate                  | Back pain relief  | Yes         |

---

Thanks for reading!
