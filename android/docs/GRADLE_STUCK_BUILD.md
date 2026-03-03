# If `./gradlew assembleDebug` is stuck

1. **Stop any running Gradle**
   ```bash
   pkill -f GradleDaemon || true
   pkill -f gradle || true
   ```

2. **Run a single-threaded build** (avoids daemon/worker hangs)
   ```bash
   cd android
   ./gradlew assembleDebug --no-daemon --max-workers=1
   ```

3. **If it hangs on a specific task**, run with stacktrace to see where:
   ```bash
   ./gradlew assembleDebug --no-daemon --stacktrace
   ```
   Then interrupt (Ctrl+C) and check the last task printed.

4. **Clean and retry**
   ```bash
   ./gradlew clean --no-daemon
   ./gradlew assembleDebug --no-daemon --max-workers=1
   ```

5. **Network**: If it’s stuck downloading dependencies, ensure you’re online and not behind a firewall blocking Gradle/Google.
