import os

from lkae.utils.data_loading import AuredDataset, root_dir

file_names = ['English_train.jsonl', 'English_dev.jsonl', 'English_test.jsonl']

blacklist = []
maybelist = []

for file_name in file_names:
    issue_dict = {} # issues per id
    config = {
        'preprocess': False,
        'add_author_name': False,
        'add_author_bio': False,
        'author_info_filepath': os.path.join(root_dir, 'data', 'combined-author-data-translated.json'),
    }

    fingerprint = 'pre-' if config['preprocess'] else 'nopre-'
    fingerprint += 'nam-' if config['add_author_name'] else 'nonam-'
    fingerprint += 'bio' if config['add_author_bio'] else 'nobio'

    ds = AuredDataset(os.path.join(root_dir, 'data', file_name), **config)

    for rumor in ds:
        id = rumor['id']
        timeline = rumor['timeline']
        total_issues = 0

        for tweet in timeline:
            if "ISSUE: couldn't translate" in tweet[2]:
                total_issues += 1

        # filter down to tweets with transl issues
        if total_issues > 0:
            
            has_evidence = False
            # test for tweets with translation issues that have 
            if 'evidence' in rumor and rumor['evidence'] and len(rumor['evidence']) > 0:
                # print(f'rumor {id} has non-empty evidence array')
                has_evidence = True
                for ev in rumor['evidence']:

                    # any evidence with translation issues?
                    # if not, we could just cull tweets from the tl with transl issues...
                    # ... as the tweet would be verifiable without those tweets 
                    if "ISSUE: couldn't translate" in ev[2]:
                        # print(f'OH NO! transl issue in evidence for {id}')
                        pass

            # calculate % of timeline tweets that have issue
            issue_percent = round((total_issues/len(timeline))*100, 1)
            if issue_percent == 100.0:
                blacklist.append(id)
            else:
                maybelist.append(id)
            issue_dict[f"{file_name}-{id}"] = {'issue_perc': issue_percent, 'has_ev': has_evidence}

    # print(issue_dict)

# we will definitely discard all rumors that have timelines ith 100% translation issues 
# print(blacklist)


# maybe keep those rumors? it's only 6 though (over all datasets)
# print(maybelist)

# for now, we'll just cull all rumors with any transl issues
cull_list = [*blacklist, *maybelist]


