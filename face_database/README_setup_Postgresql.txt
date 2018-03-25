Adding this for anybody trying to make this type of DB on MacOSX.
also for my own future reference when I forget how to do this and need to
reinstall.
(the docker solution did not work so this is the manual solution that worked for me):

Requirements:

Need to have Xcode downloaded for Cmake
Homebrew
Starters... make sure you have postgresql downloaded for used for 'pg_config'
$ brew install postgresql

INSTRUCTIONS:
download source for postgresql:
https://ftp.postgresql.org/pub/source/v9.6.0/postgresql-9.6.0.tar.bz2
(get the correct version number ___mine is: 10.3)

unzip...

change /contrib/cube/cubedata.h to include 128 dimensions:
#define CUBE_MAX_DIM (100) -> #define CUBE_MAX_DIM (128)

//128 float for facial Encodings

Follow Directions in the 'INSTALL' file at top directory,
Follow instructions for install and for starting server:

./configure
make
su
make install
adduser postgres
mkdir /usr/local/pgsql/data
chown postgres /usr/local/pgsql/data
su - postgres
/usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data

***Start Server:
/usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data

***note
(use this command to change whoami on mac)
$ sudo su - postgres

Now we need to add the extension. go to the /contrib/ directory
follow directions in the README, we can either make all make all install
for all extensions or we can navigate to /contrib/cube/
and just:
$ make
$ make install
for this one extension

Now you want to go to your database and add the extension.
For this I just used my GUI and ran the following:

CREATE EXTENSION cube
