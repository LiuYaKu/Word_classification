import os
while True:
    import test
    os.chdir('./tools')
    import tools.tidy_data
    os.chdir('./tools')
    import tools.create_train_data
    import train
