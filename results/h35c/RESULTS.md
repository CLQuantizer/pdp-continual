## Scoring table

C4-val bits/byte. G_pipe = rand − pipeline. G_prog = rand − prog_snap. pre is 1 when mean G_pipe ≥ 0.02 at that size. bar1 is 1 when G_prog ≥ 0.5 · G_pipe. bar2 is 1 when prog_snap − rand_plus ≤ −0.01. bar3 is 1 when G_prog(medium) ≥ 0.8 · G_prog(small), means over seeds. If pre is 0 at either size, bars are void.

| size | seed | rand | pipeline | prog_snap | rand_plus | G_pipe | G_prog | G_prog/G_pipe | prog_snap−rand_plus | bar1 | bar2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 0 | 2.402307 | 2.089058 | 2.492105 | 2.066538 | 0.313250 | -0.089798 | -0.286667 | 0.425568 | 0 | 0 |
| small | 1 | 2.324876 | 2.150052 | 2.517743 | 2.088654 | 0.174824 | -0.192867 | -1.103208 | 0.429089 | 0 | 0 |
| medium | 0 | 2.048373 | 1.909019 | 2.060834 | 1.904742 | 0.139354 | -0.012461 | -0.089417 | 0.156091 | 0 | 0 |
| medium | 1 | 2.044015 | 1.928200 | 2.094360 | 1.913531 | 0.115815 | -0.050346 | -0.434712 | 0.180830 | 0 | 0 |

small mean G_pipe 0.244037 mean G_prog -0.141333 pre 1 n 2
medium mean G_pipe 0.127584 mean G_prog -0.031403 pre 1 n 2
precondition 1. bars scored. G_prog(medium)/G_prog(small) 0.222194.
bar1 count 0/4. bar2 count 0/4. bar3 count 1/1.

# H35c web curiosity

S = 3000. seeds = {"small": [0, 1], "medium": [0, 1]}. cuts = [].
autocast = fp16. classifier = fp32. grad_clip = 1.0. init = N(0, 0.02), residual out and down scaled by 0.02/sqrt(2*layers).
online peek bytes = 128. online candidates = 96. rand_plus max epochs = 2.
enwik8 first 10MB sha256 = `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
pool sha256 = `5febac895e9db86fda81d77ba2e952a818a5294788f9bf51a99b9ffc97907b1e`.

## C4-val and WikiText bits/byte

| arm | size | seed | c4_bpb | wiki_bpb | steps |
|---|---|---:|---:|---:|---:|
| rand | small | 0 | 2.402307 | 2.767410 | 3000 |
| heur | small | 0 | 2.134433 | 2.498772 | 3000 |
| dedup | small | 0 | 2.413132 | 2.768828 | 3000 |
| ppl | small | 0 | 2.103174 | 2.475128 | 3000 |
| cls | small | 0 | 2.338112 | 2.696295 | 3000 |
| pipeline | small | 0 | 2.089058 | 2.471268 | 3000 |
| high_loss | small | 0 | 2.628902 | 2.978013 | 3000 |
| rho | small | 0 | 2.480244 | 2.793778 | 3000 |
| prog_snap | small | 0 | 2.492105 | 2.817897 | 3000 |
| rand_plus | small | 0 | 2.066538 | 2.414750 | 4441 |
| rand | small | 1 | 2.324876 | 2.657816 | 3000 |
| heur | small | 1 | 2.145017 | 2.532867 | 3000 |
| dedup | small | 1 | 2.438003 | 2.813551 | 3000 |
| ppl | small | 1 | 2.131982 | 2.513665 | 3000 |
| cls | small | 1 | 2.266920 | 2.635989 | 3000 |
| pipeline | small | 1 | 2.150052 | 2.539260 | 3000 |
| high_loss | small | 1 | 2.589503 | 2.912822 | 3000 |
| rho | small | 1 | 2.459059 | 2.802702 | 3000 |
| prog_snap | small | 1 | 2.517743 | 2.870539 | 3000 |
| rand_plus | small | 1 | 2.088654 | 2.431003 | 4441 |
| rand | medium | 0 | 2.048373 | 2.384016 | 3000 |
| heur | medium | 0 | 1.936368 | 2.295251 | 3000 |
| dedup | medium | 0 | 2.044682 | 2.365252 | 3000 |
| ppl | medium | 0 | 1.917150 | 2.282826 | 3000 |
| cls | medium | 0 | 1.997068 | 2.328664 | 3000 |
| pipeline | medium | 0 | 1.909019 | 2.242215 | 3000 |
| high_loss | medium | 0 | 2.119535 | 2.466467 | 3000 |
| rho | medium | 0 | 2.097096 | 2.431136 | 3000 |
| prog_snap | medium | 0 | 2.060834 | 2.371734 | 3000 |
| rand_plus | medium | 0 | 1.904742 | 2.244670 | 4466 |
| rand | medium | 1 | 2.044015 | 2.371337 | 3000 |
| heur | medium | 1 | 1.946588 | 2.307029 | 3000 |
| dedup | medium | 1 | 2.046723 | 2.381991 | 3000 |
| ppl | medium | 1 | 1.923272 | 2.292755 | 3000 |
| cls | medium | 1 | 1.978885 | 2.313043 | 3000 |
| pipeline | medium | 1 | 1.928200 | 2.273540 | 3000 |
| high_loss | medium | 1 | 2.091209 | 2.417040 | 3000 |
| rho | medium | 1 | 2.069638 | 2.408661 | 3000 |
| prog_snap | medium | 1 | 2.094360 | 2.443837 | 3000 |
| rand_plus | medium | 1 | 1.913531 | 2.261445 | 4466 |

## rand_plus sequences re-read

| size | seed | sequences_reread | epochs |
|---|---:|---:|---:|
| small | 0 | 0 | 1 |
| small | 1 | 0 | 1 |
| medium | 0 | 0 | 1 |
| medium | 1 | 0 | 1 |

## Compute

train_flops is completed steps times the measured train-step FLOPs at that size. select_flops counts scoring forwards, reference training, and classifier inference. shared passes ran once; smoke kernel holds their GPU wall; MinHash FLOPs are 0.

| arm | size | seed | train_flops | select_flops | select_pct_of_train | wall_s | wall_select_s | wall_eval_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rand | small | 0 | 1307817541632000.000 | 0.000 | 0.0000 | 138.725 | 0.000 | 3.297 |
| heur | small | 0 | 1307817541632000.000 | 0.000 | 0.0000 | 138.963 | 0.000 | 3.297 |
| dedup | small | 0 | 1307817541632000.000 | 0.000 | 0.0000 | 138.853 | 0.000 | 3.292 |
| ppl | small | 0 | 1307817541632000.000 | 1938696262123520.000 | 148.2390 | 139.058 | 0.000 | 3.297 |
| cls | small | 0 | 1307817541632000.000 | 5798276720640000.000 | 443.3552 | 139.018 | 0.000 | 3.299 |
| pipeline | small | 0 | 1307817541632000.000 | 3262593672758784.000 | 249.4686 | 139.151 | 0.000 | 3.296 |
| high_loss | small | 0 | 1307817541632000.000 | 314069483520000.000 | 24.0148 | 171.555 | 37.558 | 3.280 |
| rho | small | 0 | 1307817541632000.000 | 1064078147584000.000 | 81.3629 | 205.716 | 73.275 | 3.298 |
| prog_snap | small | 0 | 1307817541632000.000 | 628138967040000.000 | 48.0296 | 285.930 | 155.266 | 3.314 |
| rand_plus | small | 0 | 1936005900795904.000 | 0.000 | 0.0000 | 206.584 | 0.000 | 3.290 |
| rand | small | 1 | 1307817541632000.000 | 0.000 | 0.0000 | 164.733 | 0.000 | 3.947 |
| heur | small | 1 | 1307817541632000.000 | 0.000 | 0.0000 | 164.300 | 0.000 | 3.909 |
| dedup | small | 1 | 1307817541632000.000 | 0.000 | 0.0000 | 164.009 | 0.000 | 3.883 |
| ppl | small | 1 | 1307817541632000.000 | 1938696262123520.000 | 148.2390 | 164.185 | 0.000 | 3.932 |
| cls | small | 1 | 1307817541632000.000 | 5798276720640000.000 | 443.3552 | 165.053 | 0.000 | 3.896 |
| pipeline | small | 1 | 1307817541632000.000 | 3262593672758784.000 | 249.4686 | 164.392 | 0.000 | 3.877 |
| high_loss | small | 1 | 1307817541632000.000 | 314069483520000.000 | 24.0148 | 201.309 | 44.636 | 3.894 |
| rho | small | 1 | 1307817541632000.000 | 1064078147584000.000 | 81.3629 | 239.496 | 86.601 | 3.878 |
| prog_snap | small | 1 | 1307817541632000.000 | 628138967040000.000 | 48.0296 | 327.841 | 178.598 | 3.927 |
| rand_plus | small | 1 | 1936005900795904.000 | 0.000 | 0.0000 | 241.730 | 0.000 | 3.897 |
| rand | medium | 0 | 8903467204608000.000 | 0.000 | 0.0000 | 764.217 | 0.000 | 18.031 |
| heur | medium | 0 | 8903467204608000.000 | 0.000 | 0.0000 | 764.727 | 0.000 | 18.121 |
| dedup | medium | 0 | 8903467204608000.000 | 0.000 | 0.0000 | 763.090 | 0.000 | 17.866 |
| ppl | medium | 0 | 8903467204608000.000 | 1938696262123520.000 | 21.7746 | 762.857 | 0.000 | 18.092 |
| cls | medium | 0 | 8903467204608000.000 | 5798276720640000.000 | 65.1238 | 762.958 | 0.000 | 18.094 |
| pipeline | medium | 0 | 8903467204608000.000 | 3262593672758784.000 | 36.6441 | 765.026 | 0.000 | 18.044 |
| high_loss | medium | 0 | 8903467204608000.000 | 2174327193600000.000 | 24.4211 | 944.243 | 173.698 | 18.056 |
| rho | medium | 0 | 8903467204608000.000 | 2924335857664000.000 | 32.8449 | 973.608 | 217.164 | 17.886 |
| prog_snap | medium | 0 | 8903467204608000.000 | 4348654387200000.000 | 48.8423 | 1197.343 | 428.408 | 17.907 |
| rand_plus | medium | 0 | 13254294845259776.000 | 0.000 | 0.0000 | 1139.007 | 0.000 | 17.976 |
| rand | medium | 1 | 8903467204608000.000 | 0.000 | 0.0000 | 850.894 | 0.000 | 20.304 |
| heur | medium | 1 | 8903467204608000.000 | 0.000 | 0.0000 | 850.680 | 0.000 | 20.277 |
| dedup | medium | 1 | 8903467204608000.000 | 0.000 | 0.0000 | 850.311 | 0.000 | 20.246 |
| ppl | medium | 1 | 8903467204608000.000 | 1938696262123520.000 | 21.7746 | 849.581 | 0.000 | 20.110 |
| cls | medium | 1 | 8903467204608000.000 | 5798276720640000.000 | 65.1238 | 850.983 | 0.000 | 20.307 |
| pipeline | medium | 1 | 8903467204608000.000 | 3262593672758784.000 | 36.6441 | 850.302 | 0.000 | 20.195 |
| high_loss | medium | 1 | 8903467204608000.000 | 2174327193600000.000 | 24.4211 | 1052.910 | 197.271 | 20.324 |
| rho | medium | 1 | 8903467204608000.000 | 2924335857664000.000 | 32.8449 | 1091.518 | 242.535 | 20.324 |
| prog_snap | medium | 1 | 8903467204608000.000 | 4348654387200000.000 | 48.8423 | 1574.921 | 717.255 | 20.293 |
| rand_plus | medium | 1 | 13254294845259776.000 | 0.000 | 0.0000 | 1265.005 | 0.000 | 20.250 |

ref_train_flops 435939180544000.000. ppl_score_flops 1502757081579520.000. ppl_tokens 176386564. cls_full_pool_flops 5798276720640000.000. cls_pipeline_flops 3262593672758784.000 on 33761 docs. heur_wall_s 47.856. minhash_wall_s 329.589. ref_wall_s 43.566. cls_wall_s 1816.067.

## Filter survival

| filter | docs | rate | sequences |
|---|---:|---:|---:|
| heur | 34056 | 0.567600 | 213032 |
| dedup | 59010 | 0.983500 | 322541 |
| ppl | 30000 | 0.500000 | 184533 |
| cls | 30000 | 0.500000 | 183524 |
| pipeline | 16880 | 0.281333 | 111417 |
| pipe_before_cls | 33761 | 0.562683 | 211776 |

pool docs 60000 sequences 327289 need 96000.

## Overlap with pipeline survivors

| arm | size | seed | selected | shared_with_pipeline |
|---|---|---:|---:|---:|
| rand | small | 0 | 17575 | 5020 |
| heur | small | 0 | 15407 | 7619 |
| dedup | small | 0 | 17673 | 5017 |
| ppl | small | 0 | 15594 | 7287 |
| cls | small | 0 | 15667 | 8801 |
| pipeline | small | 0 | 14545 | 14545 |
| high_loss | small | 0 | 17499 | 3954 |
| rho | small | 0 | 16515 | 4833 |
| prog_snap | small | 0 | 17972 | 4838 |
| rand_plus | small | 0 | 26018 | 7391 |
| rand | small | 1 | 17595 | 4890 |
| heur | small | 1 | 15303 | 7679 |
| dedup | small | 1 | 17485 | 4904 |
| ppl | small | 1 | 15608 | 7232 |
| cls | small | 1 | 15697 | 8826 |
| pipeline | small | 1 | 14567 | 14567 |
| high_loss | small | 1 | 17497 | 3888 |
| rho | small | 1 | 16536 | 4848 |
| prog_snap | small | 1 | 17892 | 4753 |
| rand_plus | small | 1 | 26076 | 7307 |
| rand | medium | 0 | 17575 | 5020 |
| heur | medium | 0 | 15407 | 7619 |
| dedup | medium | 0 | 17673 | 5017 |
| ppl | medium | 0 | 15594 | 7287 |
| cls | medium | 0 | 15667 | 8801 |
| pipeline | medium | 0 | 14545 | 14545 |
| high_loss | medium | 0 | 17281 | 3860 |
| rho | medium | 0 | 16370 | 4765 |
| prog_snap | medium | 0 | 17651 | 4604 |
| rand_plus | medium | 0 | 26163 | 7431 |
| rand | medium | 1 | 17595 | 4890 |
| heur | medium | 1 | 15303 | 7679 |
| dedup | medium | 1 | 17485 | 4904 |
| ppl | medium | 1 | 15608 | 7232 |
| cls | medium | 1 | 15697 | 8826 |
| pipeline | medium | 1 | 14567 | 14567 |
| high_loss | medium | 1 | 17254 | 3856 |
| rho | medium | 1 | 16435 | 4737 |
| prog_snap | medium | 1 | 17680 | 4343 |
| rand_plus | medium | 1 | 26219 | 7336 |

## prog_snap examples

First selected documents, then the 10 documents rejected most often. Preview is the first 200 bytes.

### small seed 0

| kind | rank | id | count | preview |
|---|---:|---:|---:|---|
| first | 1 | 25367 |  | MRS. SARAH E. SENAVITIS - The Morning Call\nLehigh Valley Top Workplaces 2019: Winners, awards ceremony pics and more\nNews\nMRS. SARAH E. SENAVITIS\nThe Morning CallTHE MORNING CALL\nMrs. Sarah E. Senavit |
| first | 2 | 42216 |  | Cherry Wood End Tables Living Room - Wooden Home Style\nwoodhomestyle.com\nHome\nLiving Room\nCherry Wood End Tables Living Room\nCherry Wood End Tables Living Room\nLiving Room September 19, 2018 12:04\nChe |
| first | 3 | 50389 |  | Vintage - Cat Girl Porn Videos, Free Anime Xxx Videos, Hentai Sex Porn Videos\nSexy Cat Girl Porn\nVintage\nVintage Creampies #2\nvintage horny classic\nWalking around the Russian countryside\nXtime Club it |
| first | 4 | 51384 |  | Learning beyond walls: wikis, skype, & global projects (guest post) \| TeachingEnglish \| British Council \| BBC\nSkip to content\nHelp\nLog in\nSign up\nNewsletter\nLearnEnglish Teens\nLearnEnglish\nLearnEnglis |
| first | 5 | 43162 |  | /g;Zc=/"/g;ad=/'/g; bd=/\x00/g;cd=/[\x00&<>"']/;_.hd=function(a,b){var c=0;a=(0,_.Vc)(String(a)).split(".");b=(0,_.Vc)(String(b)).split(".");for(var d=Math.max(a.length,b.length),e=0;0==c&&eb?1:0}; va |
| first | 6 | 42849 |  | nice and creamy lt - Porno Pics \| prof-it2016.ru\nPorno Pics\nSearch\nExplore\nHome\nLiked\nHot\nRecent\nCategories\nAll Tags\nCollections\nSlideshow\nUpload\nCategories\n1. porn (91630)\n2. ass (83136)\n3. hentai (7 |
| first | 7 | 55697 |  | Green Solutions Carpet \| We take pride in our work\nHome\nHome Services\nCarpet Cleaning\n24 Hour Emergency Services\nCarpet Repair\nCarpet Installation\nWater damage\nWindow treatment\nRug Repair\nRug cleaning |
| first | 8 | 15520 |  | Live 5 Gaming has developed a small portfolio of innovative slot games - Casino Software \| Online Casino Reports\nToggle navigation\nNotifications\nNotification Settings\nView all notifications\nEdit Profi |
| first | 9 | 49518 |  | Opera account\nSign in\nCreate account\nSign in\nSign in with your Opera account.\nNeed an account? It's fast, easy (and free) to create an account.\nOr, sign in with\nFacebook\nTwitter\nGoogle\nVk\nSelect your  |
| first | 10 | 33230 |  | Legalize Nature Tee-Shirts \| Ink to the People \| T-Shirt Fundraising - Raise Money for Your Cause or Charity\nLoading\nCannot add this to your cart\nClose\nLookup Your Order\nOrder #: Billing Zip:\nLookup O |
| rejected | 1 | 29307 | 18 | ARCH 463 - Sustainable Systems in Architecture at the University of Maryland, College Park \| Coursicle UMD\nCoursicle\nARCH at UMD\nARCH 463 - Sustainable Systems in Architecture\nDescription\nRestriction: |
| rejected | 2 | 3468 | 17 | New 2018 Kawasaki Ninja ZX-6R KRT EDITION Motorcycles in Hicksville, NY \| Stock Number: N/A\nour hours\n67 North Broadway Route 107 \|\nHicksville, NY 11801\n(516) 935-6969\nToggle navigation\nHome\nShowroom\n |
| rejected | 3 | 55049 | 17 | Search Engine Optimization in Bournemouth - safe SEO with TopBorn\nMarketing\nSearch engine optimization – Below you can read more about\nPerformance-based SEO\nSEO Strategy\nKeyword Analysis\nCompetitive |
| rejected | 4 | 7821 | 16 | Still Life — Diana Watson\nDiana Watson\nAUSTRALIAN ARTIST\nPaintings\nFlora\nStill Life\nNova\nNew\nExhibitions 2019\nCV\nGalleries\nPress\nContact Diana Watson\nWallpaper\nLimited Edition Prints\nPaintings\nFlora |
| rejected | 5 | 12455 | 16 | Slater Flooring & Design - About Us\nHome\nAbout\nFAQ\nContact Us\nCarpet\nHardwood & Laminate\nTile & Stone\nLuxury Vinyl Tile\nKitchen & Bath Remodeling\nAbout Us\nGary Slater – President and owner of Slater |
| rejected | 6 | 21743 | 16 | Sam Cook - Technology in government\nSkip to main content\nGOV.UK blogs use cookies to make the site simpler. Find out more about cookies\nGOV.UK\nBlog Technology in government\nOrganisations:\nCivil Servic |
| rejected | 7 | 25717 | 16 | Indian Stock Market Advice for – Tuesday, June 24, 2014 \| EquityPandit\n8000816688\nInvestors\nClose\nUsername\nPassword\nLogin\nLost Password\nBlog\nInstall App\nAsk The Analyst\nTrending News\nProvision Decli |
| rejected | 8 | 35185 | 16 | Login\nThis topic This board Entire forum Google\nMultitool.org\nSAK Wiki\nMTO Encyclopedia\nMTO Gallery\nWelcome, Guest. Please login or register.\nDid you miss your activation email?\n1 Hour 1 Day 1 Week 1  |
| rejected | 9 | 1496 | 15 | Jordan Candler: Planned Parenthood Endeavors to Keep the Media Silent — The Patriot Post\nToggle navigation\nSubscribe\nSupport Us\nOur Mission\nRight Analysis\nRight Opinion\nAlexander's Columns\nTop Headl |
| rejected | 10 | 7493 | 15 | The Hacker Project - a free online hacking game\nHome\nGame guide\nGateway\nConnect\nIP Database\nManage transfers\nFactions info\nYour ability\nMission data\nSoftware\nHardware\nLogs\nGroup\nPvP Board\nGold Account |

### small seed 1

| kind | rank | id | count | preview |
|---|---:|---:|---:|---|
| first | 1 | 58855 |  | Quincy Mae Baby Booties – Magnolia Jaymes\nBlack Friday is COMING! Are you ready? sign up for the newsletter now!\nHome\nBrands +-\nFin & Vince\nFrankie Jones the Label\nGo Gently Nation\nGray Label\nIzzy a |
| first | 2 | 44981 |  | สมัครสมาชิก - casestudy.ran4u.com\n"; //btnAddToCart.innerHTML = innerBtnId; showCartDialogAfterAddSubProduct(urlParam); } else { urlParam = "addToCart.do?clubId=36725&shopCategor |
| first | 3 | 18683 |  | Gold Package – Eric Liboiron\n1-613-555-0143 ericliboiron@gmail.com\nSearch for:\nNo products in the cart.\nHome\nDeals\nDiscount Offer\nLearn More\nHow to become a speaker\n$13.00\n750 students enrolled admi |
| first | 4 | 45155 |  | Residential Cleaning Services \| SERVPRO of East Brownsville & South Padre Island\nIf you have a fire or water emergency, please call us now at (956) 747-3020\nTo have the optimal experience while using  |
| first | 5 | 25364 |  | Asian ladies like Baylee Lee love cum shots after banging all night long.\nCategories\nStreet porn\nButt fucking\nGingers\nLingerie porn\nFucking mulatto\nFetishists\nLegs\nButtocks\nPorno lovers\nSea of sperm\nS |
| first | 6 | 56940 |  | Petition · Cadbury: Stop Exploitation of African Children: Boycott Non-Fairtrade Cadbury Products · Change.org\nSkip to main content\nStart a petition\nMy petitions\nBrowse\nContributions\nLog in\nStart a  |
| first | 7 | 41889 |  | The Danger Wherein the Kingdom Now Standeth and the Remedy \| Manuscript Pamphleteering in Early Stuart England\nToggle navigation\nMPESE: Manuscript Pamphleteering in Early Stuart England\nIntroduction\nH |
| first | 8 | 59699 |  | Teeth Whitening in Derry\nSign up for the latest news and offers!\nGo\nHome\nTeeth Whitening Kits\nTeeth Whitening Strips\nTeeth Whitening Costs\nTeeth Whitening Gel\nTeeth Whitening Products\nLocations Covere |
| first | 9 | 45427 |  | DNA Today Video Collection :: DNA Learning Center\nHome DNA Learning Center\nPreparing students and families to thrive in the gene age\nprograms\nwebsites\nresources\nabout\nSpotlight Collection\nPolymerase C |
| first | 10 | 41369 |  | There are so many great apps available now to help you with everything\nHome\nDesign Services\nMeet Almar\nThe Almar Team\nEmployment Opportunities\nContact Us\nNewsroom\nClient Login\nAlmar\nBuilding and Remod |
| rejected | 1 | 8882 | 17 | Devotion to Our Lady: Queen Beauty of Carmel – Carmel of St. Therese of Lisieux\nHome Page\nOur Carmel\nThe Founding of Loretto Carmel\nThe Story of Loretto, PA\nSaint Thérèse: Our Patroness\nThe Saint  |
| rejected | 2 | 5714 | 16 | Delhi gifts, Send Delhi India\nIndia: (91 22) 4030 1121\n9:30 AM to 6 PM IST\nMy Account\nCustomer Service\nContact Us\nMy cart\nLogin\nMy Account\nLog Out\nMenu\n0 Item(s): $00.00\nToggle navigation\nHome\nCategor |
| rejected | 3 | 14699 | 16 | Audio-Technica ATH-M30 Review \| The Headphone List\nTrending\nThe Glistening of Morning Dew – A Review of the SendyAudio Aiva\nCanJam Singapore 2019 – The Hybrid Resurgence\nA Standard With and By Whi |
| rejected | 4 | 21832 | 16 | Crash of a Rockwell Aero Commander 690 in Georgia: 2 killed \| Bureau of Aircraft Accidents Archives\nSkip to main content\nBureau of Aircraft Accidents Archives\nSearch\nToggle navigation\nMain navigation\n |
| rejected | 5 | 56443 | 16 | Word Around the Net: SPRAY SHIRT\nThoughts on culture and events by author and illustrator Christopher R Taylor\nFriday, September 17, 2010\nSPRAY SHIRT\n"Paging Ms. Ines Sainz..."\nWhat's going on here?\nW |
| rejected | 6 | 59399 | 16 | Rihanna photo 7934 of 7962 pics, wallpaper - photo #1122315 - ThePlace2\nThePlace\nHome\nPhotos\nNews\nRating\nForum\nSearch\nSign in\nPhotos\nRihanna\nRihanna photo #1122315\n№ 7934 / 7461\n1\nOriginal Image siz |
| rejected | 7 | 1818 | 15 | Search 4 Barbara Court Bundoora » Darren Jones Real Estate\nBuy\nSearch Properties\nOpen for Inspection\nAuctions\nSold Properties\nFinance\nGovernment Assistance\nDue Diligence Checklist\nSell\nWhy Choose Us\n |
| rejected | 8 | 4210 | 15 | EN FR\nAbout us\nWho we are\nOur management team\nWhat makes us different\nCreating value through partnerships\nKey facts and figures\nJoin us\nNews\nExcellence in cancer care\nScientific board\nTreatments\nOur c |
| rejected | 9 | 4226 | 15 | Email Alerts - Real Estate Charlotte North Carolina\n704-322-9810 \| Winst Properties\n[email protected]\nFacebook\nTwitter\nGoogle+\nReal Estate Charlotte North Carolina\nHOME\nFEATURED\nBALLANTYNE\nSOUTH PARK\n |
| rejected | 10 | 5919 | 15 | Angry bird cake \| Cakes by Arilena\nCakes by Arilena\nIeder feest verdient een taart!\nSkip to content\nWelkom\nOver mij\nMijn Facebook pagina\nMijn taarten op Pinterest\nMijn oude website\nMijn foto’s op Fl |

### medium seed 0

| kind | rank | id | count | preview |
|---|---:|---:|---:|---|
| first | 1 | 25367 |  | MRS. SARAH E. SENAVITIS - The Morning Call\nLehigh Valley Top Workplaces 2019: Winners, awards ceremony pics and more\nNews\nMRS. SARAH E. SENAVITIS\nThe Morning CallTHE MORNING CALL\nMrs. Sarah E. Senavit |
| first | 2 | 42216 |  | Cherry Wood End Tables Living Room - Wooden Home Style\nwoodhomestyle.com\nHome\nLiving Room\nCherry Wood End Tables Living Room\nCherry Wood End Tables Living Room\nLiving Room September 19, 2018 12:04\nChe |
| first | 3 | 50389 |  | Vintage - Cat Girl Porn Videos, Free Anime Xxx Videos, Hentai Sex Porn Videos\nSexy Cat Girl Porn\nVintage\nVintage Creampies #2\nvintage horny classic\nWalking around the Russian countryside\nXtime Club it |
| first | 4 | 51384 |  | Learning beyond walls: wikis, skype, & global projects (guest post) \| TeachingEnglish \| British Council \| BBC\nSkip to content\nHelp\nLog in\nSign up\nNewsletter\nLearnEnglish Teens\nLearnEnglish\nLearnEnglis |
| first | 5 | 43162 |  | /g;Zc=/"/g;ad=/'/g; bd=/\x00/g;cd=/[\x00&<>"']/;_.hd=function(a,b){var c=0;a=(0,_.Vc)(String(a)).split(".");b=(0,_.Vc)(String(b)).split(".");for(var d=Math.max(a.length,b.length),e=0;0==c&&eb?1:0}; va |
| first | 6 | 42849 |  | nice and creamy lt - Porno Pics \| prof-it2016.ru\nPorno Pics\nSearch\nExplore\nHome\nLiked\nHot\nRecent\nCategories\nAll Tags\nCollections\nSlideshow\nUpload\nCategories\n1. porn (91630)\n2. ass (83136)\n3. hentai (7 |
| first | 7 | 49518 |  | Opera account\nSign in\nCreate account\nSign in\nSign in with your Opera account.\nNeed an account? It's fast, easy (and free) to create an account.\nOr, sign in with\nFacebook\nTwitter\nGoogle\nVk\nSelect your  |
| first | 8 | 15520 |  | Live 5 Gaming has developed a small portfolio of innovative slot games - Casino Software \| Online Casino Reports\nToggle navigation\nNotifications\nNotification Settings\nView all notifications\nEdit Profi |
| first | 9 | 10983 |  | Milton Keynes to Manchester by train \| Buy tickets from $15.71 \| Loco2\nLoco2.com uses JavaScript to bring you the best experience. Please ensure JavaScript is enabled.\nLog in / Register\nEnglish\nUSD\nDe |
| first | 10 | 3660 |  | Plan cul cougar instacougar rencontre femme mure\nXML Sitemap\nThis is a XML Sitemap which is supposed to be processed by search engines which follow the XML Sitemap standard like Ask.com, Bing, Google  |
| rejected | 1 | 30278 | 17 | Contender Fight Sports Body Protector Review - MMA Life\nHome\nBlog\nReviews\nSingle Product Reviews\nMulti-Product Reviews\nVideos\nGeneral MMA\nGrappling\nJiu Jitsu\nJudo\nStriking\nBoxing\nKickboxing\nConditioni |
| rejected | 2 | 3073 | 16 | Skateboard Policy \| California State University, Long Beach\nJump to Content Jump to Resources\nJavaScript is required to use this site.\nCSULB\nUniversity Police\nApply\nExplore\nMenu\nACADEMICS\nColleges & D |
| rejected | 3 | 6023 | 16 | Samitivej Hospital Review \| Iguzo\nMenu\nFitness Equipment\nElliptical Trainer\nExercise Bike\nExercise Machines\nHome Exercise Equipment\nMulti Gym\nTreadmill\nWeight Lifting Equipment\nHCG Diet\nDiet Drops\nHCG |
| rejected | 4 | 10090 | 16 | Global Surveyor begins long-delayed mapping of Mars » MarsNews.com\nToggle navigation\nHome\nFeatures\nThe Planet Mars\nPhobos & Deimos\nLife on Mars?\nTerraforming\nMissions\nHumans to Mars\nSample Return\nExo |
| rejected | 5 | 55795 | 16 | Advantage Controls receives exporting award – The Journal Record\nSubscribe\nManage Account\nManage Your Print or Online Subscription\nManage Your Email Subscription\nLOG IN\nThe Journal Record Your busin |
| rejected | 6 | 56378 | 16 | I was wondering if you need volunteers of extra employees in the library? I have lots of free time on weekends so I would love to help around in - LibAnswers\nSkip to Main Content\nLibrary\nLibAnswers\nWa |
| rejected | 7 | 1009 | 15 | Care Concept AG • Health insurance for students and language pupils in USA, Canada and Mexico • Care College USA • Introduction\nYour reliable\nhealth insurance partner abroad\nForeigners in Europe |
| rejected | 8 | 1429 | 15 | Office HD PORN - In the office, the boys are screwing the angels / hdporntub.com\nHome\nVideos\nCategories\nModels\nStudios\nen\nOffice HD PORN\n- In the office, the boys are screwing the angels\nSorted video  |
| rejected | 9 | 11219 | 15 | New Jaguar E-Pace compact SUV – an X1, Q3 rival Paul Tan - Image 683015\nBack to Story: New Jaguar E-Pace compact SUV – an X1, Q3 rival\nNext Image\nPrevious Image\nHome\nAbout Us\nContact Us\nAdvertise  |
| rejected | 10 | 12084 | 15 | Momscribe: parenting and kids\nPlease enable javascript to view this site.\n≡Menu\nHome\nAbout\nParenting and Kids\nToddlers / Kids\nArts and Crafts\nEducation and Learning\nApps for Kids\nKids Fashion\nKids S |

### medium seed 1

| kind | rank | id | count | preview |
|---|---:|---:|---:|---|
| first | 1 | 58855 |  | Quincy Mae Baby Booties – Magnolia Jaymes\nBlack Friday is COMING! Are you ready? sign up for the newsletter now!\nHome\nBrands +-\nFin & Vince\nFrankie Jones the Label\nGo Gently Nation\nGray Label\nIzzy a |
| first | 2 | 44981 |  | สมัครสมาชิก - casestudy.ran4u.com\n"; //btnAddToCart.innerHTML = innerBtnId; showCartDialogAfterAddSubProduct(urlParam); } else { urlParam = "addToCart.do?clubId=36725&shopCategor |
| first | 3 | 18683 |  | Gold Package – Eric Liboiron\n1-613-555-0143 ericliboiron@gmail.com\nSearch for:\nNo products in the cart.\nHome\nDeals\nDiscount Offer\nLearn More\nHow to become a speaker\n$13.00\n750 students enrolled admi |
| first | 4 | 45155 |  | Residential Cleaning Services \| SERVPRO of East Brownsville & South Padre Island\nIf you have a fire or water emergency, please call us now at (956) 747-3020\nTo have the optimal experience while using  |
| first | 5 | 25364 |  | Asian ladies like Baylee Lee love cum shots after banging all night long.\nCategories\nStreet porn\nButt fucking\nGingers\nLingerie porn\nFucking mulatto\nFetishists\nLegs\nButtocks\nPorno lovers\nSea of sperm\nS |
| first | 6 | 56940 |  | Petition · Cadbury: Stop Exploitation of African Children: Boycott Non-Fairtrade Cadbury Products · Change.org\nSkip to main content\nStart a petition\nMy petitions\nBrowse\nContributions\nLog in\nStart a  |
| first | 7 | 30016 |  | The Learning for Life Programme \| Kirkleatham Hall School\nMenu\nPlease look in our "Key Information - E-Safety" section for important information on keeping your child SAFE ON LINE.\nA Translate\nB Searc |
| first | 8 | 37930 |  | Planning for Graduate Education in Chemistry - American Chemical Society\n×\nYour browser is looking a little out of date...\nTo get the best possible experience using our website, we recommend that you |
| first | 9 | 45427 |  | DNA Today Video Collection :: DNA Learning Center\nHome DNA Learning Center\nPreparing students and families to thrive in the gene age\nprograms\nwebsites\nresources\nabout\nSpotlight Collection\nPolymerase C |
| first | 10 | 41889 |  | The Danger Wherein the Kingdom Now Standeth and the Remedy \| Manuscript Pamphleteering in Early Stuart England\nToggle navigation\nMPESE: Manuscript Pamphleteering in Early Stuart England\nIntroduction\nH |
| rejected | 1 | 15821 | 16 | doris doppler : Anna\nsearch\n.\n+\ndoris doppler\nabout\ncontact\nimprint\nAnna\ncategory portrait\nEmail This BlogThis! Share to Twitter Share to Facebook\n0 Kommentare:\nPost a Comment\n\| Next \\ // Prev \|\nRss: |
| rejected | 2 | 17807 | 16 | RIC : RADProductsOnline, Buy RAD Products Online at Great Prices\nSearch\nHome\| Reseller\|\nContact\| Cart\nSearch\nHome\nResellers\nContact\nCart\nHello Guest, Login Create New Account\nHOME\nPRODUCTS\nACE - EGATE |
| rejected | 3 | 21821 | 16 | Query about one aspect of migrating .pst files from Outlook 2003 to Outlook 2013 \| Outlook Forums by Slipstick.com\nMenu\nForums\nForum list Search forums\nWhat's new\nNew posts New resources Latest activi |
| rejected | 4 | 29496 | 16 | Why Greyhounds Are Great Pets For All Types of Families\nSkip to content\nEasy Breeze Farm\nBreeding and Training Done Right\nHorse Breeding\nHorse Training\nGreyhounds\nAbout\n8 Great Reasons to Adopt a Grey |
| rejected | 5 | 32020 | 16 | Denton \| Other \| Next weekend \| Events at wfaa.com\nApr 25\nEventsfind something to do\nSearch\nMobile Site Add An Event\nExpand Results\nCity: Denton\nVenue: Other\nDate: Next week\nDate: Next weekend\nClear A |
| rejected | 6 | 40702 | 16 | Paper Clippings – Official Lowell Sanders Web Site\nSkip to content\nHome\nCalendar\nGuestbook\nBio\nStore\nVideo\nGallery\nCelebrities\nFans\nTours\nPaper Clippings\nContacts\nBlog\nHome Gallery Paper Clippings\nP |
| rejected | 7 | 41166 | 16 | Unknown Product - Red Inc. Limited \| Red Inc. Limited\nThis website uses cookies to continuously improve the services and to enable certain features. You agree to their use by remaining on this site. O |
| rejected | 8 | 54486 | 16 | Asheville Unreported: May 2018\nAsheville Unreported\nIf they won't cover it, we will.\nHome\nAbout\nArticle Submissions\nContact\nTuesday, May 29, 2018\nCity of Asheville Mid-Year Salary Increases Report\nCit |
| rejected | 9 | 5685 | 15 | Reusable plastic bid cards for your auctions- several sizes and styles \| Better Bidders Inc\nPaddle Pricing\nPaddle Shapes\nOur Most Popular Shape\nBasic Shapes\nOvals\nCircles\nSquares\nRectangles\nStars\nDiam |
| rejected | 10 | 11795 | 15 | â˜Ž Telephones of Whores Limehill - Hookers\nSkip to content\nWHORES\nProstitutes and whores\nHome\nSEX CHAT\nTelephones of Whores Limehill\nI want real sex with hookers Limehill â€” pay from 60$ and |

## Smoke

```
gpu Tesla T4
ok 1 wall_s 2331.550746744 session_s 2387.4429869651794
eval_during_selection 0
loss_fell 1 rand_wall_s 37.030712359999995
rand_first_step 0 rand_first_bpb 8.010217167305141
rand_last_step 199 rand_last_bpb 4.01538385911102
prog_steps 401 draws 401 scored_tokens 25955098 snap_tokens 25955098
capture_n 128 mean 0.11326864671758656 std 0.17792049495181989 percentiles {"0": -0.22236156463623047, "10": 0.03946194648742676, "25": 0.07382512092590332, "50": 0.10253667831420898, "75": 0.13124406337738037, "90": 0.15538649559020995, "100": 1.9766650199890137}
capture_highest id 57176 score 1.9766650199890137 SLTB buses - ශ්‍රී ලංගම බස්: Ideas - අදහස්\nBuses of Sri Lanka Transport Board (SLTB). (formerly: Ceylon Transport Board, CTB) -- ශ්‍රී ලංගම බස�
capture_highest id 33045 score 0.34462833404541016 Being Wagner\nHome\nAnnabel Lee\nUlalume\nThumbnail Reviews\nCategories\nART\nAUSTRALIANIA\nCRAFT\nDANTE\nFILM\nHISTORY\nLESLEY’S WRITING\nLIFE\nMUSIC\nPETER’S WRITING\nPOLITICS\nTRAVEL\nVIDEO GAMES\nWAGNER\nWRITING 
capture_highest id 20294 score 0.2581338882446289 Kase Filters Wolverine ND64 100x100mm ND filter Archives - Phil Sproson Photography\nSkip to content\nFacebookInstagramTwitterGoogle+LinkedInThe Outdoor Guide\nMy Account\nRemember Me\nRegister\nCart\nHOME\nA
capture_lowest id 46516 score -0.07567214965820312 Puig Hi-Tech Parts\nAbout Puig\nNews\nPDF Catalog\nNewsletter\nVirtual wind tunnel\nMultimedia\nProducts\nPRODUCT NEWS\nDealers\nContact us\nCustomer access\nBuscar\nAbout Puig\nNews\nPDF Catalog\nNewsletter\nVirtual 
capture_lowest id 43924 score -0.12832117080688477 AstDyS\nAstDyS-2\nAsteroids - Dynamic Site\nGo to asteroid ▶\nA A A\n- - ◄ ►\n▲\n▼\nAstDyS ∴ Objects ∴ 1987SM12 ∴ Proper elements\nContact »\nHome\nObjects\nObservatories\nSearch\nOrbital elements\n
capture_lowest id 35580 score -0.22236156463623047 Omar Hammad Atassi, MD | Tampa General Hospital /g,""),i=t[r]||t._default;e=i[1]+e+i[2],n.innerHTML=e;for(var a=i[0]+1;a--;)n=n.lastChild}else n.innerHTML=e,n=n.lastChild;return n};!function(){functio
delta {"small": {"delta": 7590, "selection_flops_at_S": 3308665276737755.5, "formula_selection_flops_at_S": 3309743389016064.0, "formula_delta": 7593, "train_step_flops": 435939180544.0, "forward_flops_per_token": 8519680.0, "measured_tokens_per_step": 129451.86034912718, "formula_tokens_per_step": 129494.0416, "draws_per_step": 1.0}, "medium": {"delta": 7719, "selection_flops_at_S": 2.2906144223569076e+16, "formula_selection_flops_at_S": 2.291360807780352e+16, "formula_delta": 7721, "train_step_flops": 2967822401536.0, "forward_flops_per_token": 58982400.0, "measured_tokens_per_step": 129451.86034912718, "formula_tokens_per_step": 129494.0416, "draws_per_step": 1.0}}
seconds_per_step {"small/rand": 0.040245973368202535, "small/high_loss": 0.09112744847759277, "small/rho": 0.14394409324999913, "small/prog_snap": 0.14505871675000037, "medium/rand": 0.19630860099999836, "medium/high_loss": 0.44449420466668244, "medium/rho": 0.5003597189999974, "medium/prog_snap": 0.7075558463333399}
seconds_per_step_raw {"small/rand": 2.496655855500002, "small/high_loss": 6.004072211249991, "small/rho": 0.14394409324999913, "small/prog_snap": 0.14505871675000037, "medium/rand": 0.19630860099999836, "medium/high_loss": 0.44449420466668244, "medium/rho": 0.5003597189999974, "medium/prog_snap": 0.7075558463333399}
timing_correction {"prog_ratio": 0.20501380562639163, "replaced": {"small/rand": {"measured": 2.496655855500002, "used": 0.040245973368202535}, "small/high_loss": {"measured": 6.004072211249991, "used": 0.09112744847759277}}}
Selection scores a candidate with the current weights, and for prog_snap also with the stored snapshot. No evaluation bytes are read.
cpu_preview_seed 0 pool_docs 60000 pool_bytes 176386564
cpu_doc id 38217 line 39506 bytes 3162 ﻿﻿ 008115 | Cantus Index for Office and Mass\nSkip to main content\nAboutInformation\nNews\nContributors\nFAQ\nTutorial\nMerged Cantus IDs\nChantsCatalogue of texts\nAnalecta Hymnica\nAnalysis Tool (BETA)\nM
cpu_doc id 30668 line 31693 bytes 4096 Warning: Use of undefined constant PHP - assumed 'PHP' (this will throw an Error in a future version of PHP) in /var/www/fs1/9/leuk/public_html/system/pre.inc.php on line 159\nWarning: Use of undefined
cpu_doc id 51035 line 52765 bytes 4096 Michael Powell | scarletthefilmmagazine Michael Powell – scarletthefilmmagazine\nSearch\nRecent Posts\nBEDAZZLED (Twilight Time Blu Ray) Fox 1967\nBLUE MOVIE(1971) Cult Epics Blu Ray/DVD combo release\nM
heur 34056/60000 seq 213032 dedup 59010 seq 322541 pipe_before_cls 33761 seq 211776
```

## Kernels

| id | state | session_h | est_h | attempts |
|---|---|---:|---:|---:|
| ezzzzzx/h35c-smoke | COMPLETE | 0.6632 |  | 1 |
| ezzzzzx/h35c-smoke | COLD_GATE | 0.0797 |  | 1 |
| ezzzzzx/h35c-m00 | COMPLETE | 3.4628 | 2.1256 | 2 |
| ezzzzzx/h35c-m10 | COMPLETE | 2.8800 | 2.1256 | 1 |
| ezzzzzx/h35c-s00 | COMPLETE | 0.5003 | 0.4461 | 1 |
| ezzzzzx/h35c-s10 | COMPLETE | 0.5826 | 0.4461 | 1 |

gpu_hours 8.1686. plan_hours 5.8065. longest_arm_h 0.3077.
