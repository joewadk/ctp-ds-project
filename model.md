# Trying out the model
Be sure to `pip install -r requirements` before proceeding. 
<br>
Now run
```bash
python capture_sign.py
```
This will allow you to capture your own gesture and stores it within `landmarks.parquet`. To stop recording press `q` on your keyboard.

Now lets run 
```bash
python classifier.py
```
This will show in console the predicted gesture in english text. Its not 100% accurate but its pretty good.

# Next steps/ Envisioned Pipeline
We want to call the function `capture_sign()` within `capture_sign.py` as soon as a user click start. We will have two buttons on screen `next` and `end`. On click of `next` we stop recording and under the hood we call `classifier()` within `classifier.py` to get the english text. Then, still under the hood, we call argos translate and using the specified language in the drop down we use argos translate to get that specific translated text. Now we want to push this translated text into Yared's text to speech model and store within a mp3. On streamlit we can show a loading screen while this is all rendering (shouldnt be that long of a wait though) and a user can press a button `Hear` to hear what the output mp3 has.