# Why does my code run everything all the time?

## The Streamlit Execution Model
Streamlit works differently than standard Python scripts. 

**Every time you interact with a widget (like a slider, a button, or a text input), Streamlit re-runs your ENTIRE `main.py` script from top to bottom.**

### Why?
This makes it easy to write apps without managing complex state updates. You just write the script as if it runs once, and Streamlit handles the updates.

### The Consequence
1.  **Count Reset:** If you have `count = 0` at the top of your script, it resets to 0 every time you click a button.
2.  **Performance:** If you have heavy computations (like loading a model), they would run every time.
    *   **Solution:** That is why we use `@st.cache_resource`. It tells Streamlit: *"If this function has already run, don't run it again. Just return the saved result."*

## Your Memory Issue
You currently have **8+ copies** of your app running in the background. This is why your computer is crashing (`OSError: paging file too small`).
I am running a command to kill them all. After that, you should run `streamlit run main.py` **only once**.
