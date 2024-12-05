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
