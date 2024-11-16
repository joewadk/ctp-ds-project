## Loading new gestures

i already have a basic model in `action.h5` with simple gesture recognition for `hello` `thank you` and `i love you`. Unfortunately it is not eady to add gestures on top of an existing model (or there is but i dont have the brain capacity for it)

if you want to load more gestures, here's a few things to note. 

- you need a `/data` folder in root. this is where gestures go.
- go into `new_gesture.py`, and *add on* to the np array `actions` of gesture with words. ill provide a list of words later.
- now run `py new_gesture.py` in the terminal. this will open up a new python window that will tell you it is recording frame. it is VERY IMPORTANT to 1) know the gesture ur doing in american sign language (google lol), 2) the frames NEED to be varied (turn slightly or move your hands closer or further from the screen), and 3) for directional words, simply just use the word 'i' or 'me' instead.
- once ur done training, it'll run 1000 epochs (training rounds) for the gestures. itll output a model called `new_actions.h5`
- now go into `mediapipe_hands.py` and make `actions` to the same as the `actions` you defined in `new_gesture.py`
- replace inside `load_model()` `action.h5` with `new_action.h5`
- with that you can test ur currently defined model via `py mediapipe_hands.py` inside terminal. GG!
- if your current model doesn't work correctly, then delete the contents of `/data` (NOTE: do not delete data. just the folders inside) then run `new_gesture.py` again
<br>

## A list of words I would like to see implemented:

- [ ] I
- [ ] me
- [ ] can
- [ ] help
- [ ] computer
- [ ] science 
- [ ] am
- [ ] where
- [ ] closest
- [ ] bathroom
- [ ] restaurant


