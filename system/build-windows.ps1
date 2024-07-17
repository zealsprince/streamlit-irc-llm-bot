cxfreeze -c run.py --target-dir $args[0]

Copy-Item app.py $args[0]

Copy-Item -Recurse system/resources/* $args[0]
