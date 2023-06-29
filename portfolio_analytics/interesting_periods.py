import pandas as pd


PERIODS = dict()


PERIODS["The Great Crash"] = (pd.Timestamp("1929-10-24"), pd.Timestamp("1933-03-15"))
PERIODS["Recession of 1937-1938"] = (
    pd.Timestamp("1937-06-01"),
    pd.Timestamp("1938-07-01"),
)

PERIODS["Kennedy Slide of 1962"] = (
    pd.Timestamp("1961-12-01"),
    pd.Timestamp("1962-07-01"),
)

PERIODS["Black Monday"] = (pd.Timestamp("1987-10-19"), pd.Timestamp("1987-10-20"))
PERIODS["Black Monday Aftermath"] = (
    pd.Timestamp("1987-10-19"),
    pd.Timestamp("1988-01-01"),
)

PERIODS["Friday the 13th"] = (pd.Timestamp("1989-10-13"), pd.Timestamp("1989-10-13"))
PERIODS["1990s Recession"] = (pd.Timestamp("1990-07-01"), pd.Timestamp("1991-04-01"))

PERIODS["Japanese Lost Decades"] = (
    pd.Timestamp("1991-01-01"),
    pd.Timestamp("2011-01-01"),
)
PERIODS["UK Black Wednesday"] = (pd.Timestamp("1992-09-16"), pd.Timestamp("1992-11-01"))
PERIODS["Asian Financial Crisis"] = (
    pd.Timestamp("1997-07-02"),
    pd.Timestamp("1999-01-01"),
)
PERIODS["Russian Financial Crisis"] = (
    pd.Timestamp("1998-08-17"),
    pd.Timestamp("2000-01-01"),
)

PERIODS["Dotcom Crash"] = (pd.Timestamp("2000-03-10"), pd.Timestamp("2000-11-09"))
PERIODS["9/11"] = (pd.Timestamp("2001-09-11"), pd.Timestamp("2001-10-04"))

PERIODS["Housing Market Boom"] = (
    pd.Timestamp("2004-06-30"),
    pd.Timestamp("2006-06-29"),
)
PERIODS["Low Volatility Bull Market"] = (
    pd.Timestamp("2005-01-01"),
    pd.Timestamp("2007-08-01"),
)

PERIODS["Chinese Bubble of 2007"] = (
    pd.Timestamp("2007-02-27"),
    pd.Timestamp("2008-01-01"),
)

PERIODS["Lehman"] = (pd.Timestamp("2008-08-15"), pd.Timestamp("2008-10-01"))
PERIODS["GFC Crash"] = (pd.Timestamp("2007-08-01"), pd.Timestamp("2009-04-01"))
PERIODS["GFC Recovery"] = (pd.Timestamp("2009-04-01"), pd.Timestamp("2013-01-01"))

PERIODS["August 2007"] = (pd.Timestamp("2007-08-01"), pd.Timestamp("2007-09-01"))
PERIODS["March 2008"] = (pd.Timestamp("2008-03-01"), pd.Timestamp("2008-04-01"))
PERIODS["September 2008"] = (pd.Timestamp("2008-09-01"), pd.Timestamp("2008-10-01"))
PERIODS["2009Q1"] = (pd.Timestamp("2009-01-01"), pd.Timestamp("2009-03-01"))
PERIODS["2009Q2"] = (pd.Timestamp("2009-03-01"), pd.Timestamp("2009-06-01"))

PERIODS["Flash Crashof 2010"] = (pd.Timestamp("2010-05-05"), pd.Timestamp("2010-05-07"))

PERIODS["Fukushima"] = (pd.Timestamp("2011-03-11"), pd.Timestamp("2011-04-11"))

PERIODS["European Debt Crisis"] = (
    pd.Timestamp("2009-10-01"),
    pd.Timestamp("2010-06-01"),
)
PERIODS["US downgrade"] = (
    pd.Timestamp("2011-08-02"),
    pd.Timestamp("2011-09-12"),
)
PERIODS["ECB Euro Announcements"] = (
    pd.Timestamp("2012-07-26"),
    pd.Timestamp("2012-10-01"),
)
PERIODS["Taper Tantrum"] = (pd.Timestamp("2013-05-21"), pd.Timestamp("2013-09-10"))

PERIODS["April 2014"] = (pd.Timestamp("2014-04-01"), pd.Timestamp("2014-05-01"))
PERIODS["October 2014"] = (pd.Timestamp("2014-10-01"), pd.Timestamp("2014-11-01"))
PERIODS["Fall 2015"] = (pd.Timestamp("2015-08-15"), pd.Timestamp("2015-09-30"))

PERIODS["Chinese Turbulence 2015-2016"] = (
    pd.Timestamp("2015-06-12"),
    pd.Timestamp("2016-02-01"),
)

PERIODS["Returning to Normalcy"] = (
    pd.Timestamp("2015-12-17"),
    pd.Timestamp("2018-12-20"),
)

PERIODS["COVID Pandemic"] = (pd.Timestamp("2020-03-11"), pd.Timestamp("2023-05-05"))
PERIODS["COVID Market Crash"] = (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-07"))
PERIODS["COVID Rate Cuts"] = (pd.Timestamp("2020-03-03"), pd.Timestamp("2020-03-16"))
PERIODS["COVID Bull Market"] = (pd.Timestamp("2020-08-01"), pd.Timestamp("2021-12-31"))

PERIODS["2022-2023 Rate Hikes"] = (
    pd.Timestamp("2022-03-17"),
    pd.Timestamp("2023-05-03"),
)

PERIODS["Cryptocurrency Crash"] = (
    pd.Timestamp("2021-04-01"),
    pd.Timestamp("2023-06-01"),
)
