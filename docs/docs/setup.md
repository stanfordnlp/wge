# Setup

## Dev Setup

* Python dependencies
  ```
  pip install -r requirements.txt
  ```
  (requirements.txt may be out of date)
* Selenium
  - Outside this repository, download [ChromeDriver](https://sites.google.com/a/
    chromium.org/chromedriver/downloads). Unzip it and then add the directory
    containing the `chromedriver` executable to the `PATH` environment variable
    ```
    export PATH=/path/to/chromedriver
    ```
* MiniWoB
  * There are 2 ways to access MiniWoB tasks:
    * **Use the `file://` protocol (Recommended):**
        Open `miniwob-sandbox/html/` in the browser,
        and then export the URL to the `MINIWOB_BASE_URL` environment variable:
        ```
        export MINIWOB_BASE_URL='file:///path/to/miniwob-sandbox/html/'
        ```
    * **Run a simple server:** go to `miniwob-sandbox/html/` and run `http-serve`.
        * The tasks should now be accessible at `http://localhost:8000/miniwob/`
        * To use a different port (say 8765), run `http-serve -p 8765`, and then
        export the following to the `MINIWOB_BASE_URL` environment variable:
        ```
        export MINIWOB_BASE_URL='http://localhost:8765/'
        ```
    * Test `MiniWoBEnvironment` by running
      ```
      pytest variational/tests/miniwob/test_environment.py -s
      ```

## Quick start

If you just want to see something happen:
```
python main.py configs/miniwob/debug-base.txt configs/miniwob/task-mixins/click-button.txt
```
* This executes the main entrypoint script, `main.py`. In particular, we pass it
  two config files.
* Both config files are specified in HOCON format, and are merged into a single
  config by main.py

If the script is working, you should see several Chrome windows pop up 
(operated by Selenium) and a training progress bar in the terminal.
