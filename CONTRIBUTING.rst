.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/hyriver/pynldas2/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Other than new features that you might have in mind, you can look through
the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

AsyncRetriever could always use more documentation, whether as part of the
official AsyncRetriever docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/hyriver/pynldas2/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up AsyncRetriever for local development.

1. Fork the AsyncRetriever repo through the GitHub website.
2. Clone your fork locally and add the main AsyncRetriever as the upstream remote:

.. code-block:: console

    $ git clone git@github.com:your_name_here/pynldas2.git
    $ git remote add upstream git@github.com:hyriver/pynldas2.git

3. Install your local copy into a virtualenv. Assuming you have Conda installed, this is how you
   can set up your fork for local development:

.. code-block:: console

    $ cd pynldas2/
    $ conda env create -f ci/requirements/environment-dev.yml
    $ conda activate pynldas2-dev

4. Create a branch for local development:

.. code-block:: console

    $ git checkout -b bugfix-or-feature/name-of-your-bugfix-or-feature
    $ git push

5. Before you first commit, pre-commit hooks needs to be setup:

.. code-block:: console

    $ pre-commit install
    $ pre-commit run --all-files

6. Now you can make your changes locally, make sure to add a description of
   the changes to ``HISTORY.rst`` file and add extra tests, if applicable,
   to ``tests`` folder. Also, make sure to give yourself credit by adding
   your name at the end of the item(s) that you add in the history like this
   ``By `Taher Chegini <https://github.com/hyriver>`_``. Then,
   fetch the latest updates from the remote and resolve any merge conflicts:

.. code-block:: console

    $ git fetch upstream
    $ git merge upstream/name-of-your-branch

7. Run the tests using ``nox``. Before running ``nox``, make sure that Python 3.9
   is installed on your system. For example, on Ubuntu-based systems you can
   install them using ``sudo apt install python3.9 python3.9-dev``:

.. code-block:: console

    $ nox

8. If you are making breaking changes make sure to reflect them in
   the documentation, ``README.rst``, and tests if necessary.

9. Commit your changes and push your branch to GitHub:

.. code-block:: console

    $ git add .
    $ git commit -m "Detailed description of your changes."
    $ git push origin name-of-your-branch

10. Submit a pull request through the GitHub website.

Tips
----

To run a subset of tests:

.. code-block:: console

    $ pytest -k "test_name1 or test_name2"

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in ``HISTORY.rst``).
Then run:

.. code-block:: console

    $ git tag -a vX.X.X -m "vX.X.X"
    $ git push --follow-tags

where ``X.X.X`` is the version number following the
`semantic versioning spec <https://semver.org>`__ i.e., MAJOR.MINOR.PATCH.
Then release the tag from Github and Github Actions will deploy it to PyPi.
