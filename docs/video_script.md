# Video Script (simple version, ~2 minutes)

Before recording:
- Run `python app/app.py` and open http://127.0.0.1:7860
- Model: "BERTurk · INT8 · ONNX Runtime"
- Open the slides in a second tab
- Start recording: Cmd + Shift + 5 (turn the microphone on)

---

## 1. Cover slide (0:00 – 0:15)

> Hi, I'm Elif Esma Doğruer.
> My project is real-time chat moderation for Turkish.
> The app finds offensive messages and hides them.

## 2. App demo (0:15 – 1:10)

> This is my app. It uses a small, fast BERTurk model on the CPU.

**Type:** `bu akşam maç kaç saatte başlıyor?`

> This message is clean.
> On the right, you can see the time. It is under 50 milliseconds.

**Type:** `s4l4k misin sen, oyunu bilmiyorsun`

> This message is offensive. The bad word is hidden.

**Type:** `senin gibi beyni yarım adamlarla oynamak zorunda kalıyoruz`

> This one is harder. There is no bad word, but it is still an insult.
> The model finds it, and the whole message is hidden.

## 3. Architecture slide — slide 4 (1:10 – 1:30)

> This is how it works.
> First, the message is cleaned.
> Then the model gives a score.
> If the score is high, the message is marked as offensive.

## 4. Results slide — slide 9 (1:30 – 1:50)

> I tested four models on the same test data.
> BERTurk is the best, with a score of 0.81.
> After quantization, it is four times smaller and still fast enough.

## 5. End (1:50 – 2:00)

> Thank you for watching.
