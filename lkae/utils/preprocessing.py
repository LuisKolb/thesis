import re
import numpy as np

def clean_tweet_basic(text):
    # source: https://github.com/Fatima-Haouari/AuFIN/blob/main/code/utils.py
    if (text is None) or (text is np.nan):
        return ""
    else:
        text = re.sub(r"http\S+", " ", text)  # remove urls
        text = re.sub(r"RT ", " ", text)  # remove rt
        text = re.sub(r"@[\w]*", " ", text)  # remove handles
        text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)  # remove special characters
        text = re.sub(r"\t", " ", text)  # remove tabs
        text = re.sub(r"\n", " ", text)  # remove line jump
        text = re.sub(r"\s+", " ", text)  # remove extra white space
        text = re.sub(r"\u201c", " ", text)  # remove “ character
        text = re.sub(r"\u201d", " ", text)  # remove “ character
        accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove

        arabic_punc= re.compile(r'[\u0621-\u063A\u0641-\u064A\d+]+') # Keep only Arabic letters/do not remove numbers
        text=' '.join(arabic_punc.findall(accents.sub('',text)))
        text = text.strip()
        return text
    
def clean_tweet_aggressive(text):
    if (text is None) or (text is np.nan):
        return ""
    else:
        text = re.sub(r"http\S+", " ", text)  # remove urls
        text = re.sub(r"RT ", " ", text)  # remove rt
        text = re.sub(r"@[\w]*", " ", text)  # remove handles
        
        # TODO: does this improve results? hashtags are... whacky; see e.g. rumor AuRED_104
        text = re.sub(r"[\.\,\|\:\?\?\/\=]", " ", text)  # remove special characters

        text = re.sub(r"#[\w]*", " ", text)  # remove hashtags

        text = re.sub(r"\t", " ", text)  # remove tabs
        text = re.sub(r"\n", " ", text)  # remove line jump
        text = re.sub(r"\s+", " ", text)  # remove extra white space
        text = re.sub(r"\u201c", "", text)  # remove “ character
        text = re.sub(r"\u201d", "", text)  # remove ” character
        text = re.sub(r"\u2018", "", text)  # remove ‘ character
        text = re.sub(r"\u2019", "", text)  # remove ’ character
        text = re.sub(r'\"', " ", text)  # remove " character

        # see: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
        emojis = r"(?:[0-9#*]️⃣|[☝✊-✍🎅🏂🏇👂👃👆-👐👦👧👫-👭👲👴-👶👸👼💃💅💏💑💪🕴🕺🖐🖕🖖🙌🙏🛀🛌🤌🤏🤘-🤟🤰-🤴🤶🥷🦵🦶🦻🧒🧓🧕🫃-🫅🫰🫲-🫸][🏻-🏿]?|⛓(?:️‍💥)?|[⛹🏋🏌🕵](?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|❤(?:️‍[🔥🩹])?|🇦[🇨-🇬🇮🇱🇲🇴🇶-🇺🇼🇽🇿]|🇧[🇦🇧🇩-🇯🇱-🇴🇶-🇹🇻🇼🇾🇿]|🇨[🇦🇨🇩🇫-🇮🇰-🇵🇷🇺-🇿]|🇩[🇪🇬🇯🇰🇲🇴🇿]|🇪[🇦🇨🇪🇬🇭🇷-🇺]|🇫[🇮-🇰🇲🇴🇷]|🇬[🇦🇧🇩-🇮🇱-🇳🇵-🇺🇼🇾]|🇭[🇰🇲🇳🇷🇹🇺]|🇮[🇨-🇪🇱-🇴🇶-🇹]|🇯[🇪🇲🇴🇵]|🇰[🇪🇬-🇮🇲🇳🇵🇷🇼🇾🇿]|🇱[🇦-🇨🇮🇰🇷-🇻🇾]|🇲[🇦🇨-🇭🇰-🇿]|🇳[🇦🇨🇪-🇬🇮🇱🇴🇵🇷🇺🇿]|🇴🇲|🇵[🇦🇪-🇭🇰-🇳🇷-🇹🇼🇾]|🇶🇦|🇷[🇪🇴🇸🇺🇼]|🇸[🇦-🇪🇬-🇴🇷-🇹🇻🇽-🇿]|🇹[🇦🇨🇩🇫-🇭🇯-🇴🇷🇹🇻🇼🇿]|🇺[🇦🇬🇲🇳🇸🇾🇿]|🇻[🇦🇨🇪🇬🇮🇳🇺]|🇼[🇫🇸]|🇽🇰|🇾[🇪🇹]|🇿[🇦🇲🇼]|🍄(?:‍🟫)?|🍋(?:‍🟩)?|[🏃🚶🧎](?:‍(?:[♀♂]️(?:‍➡️)?|➡️)|[🏻-🏿](?:‍(?:[♀♂]️(?:‍➡️)?|➡️))?)?|[🏄🏊👮👰👱👳👷💁💂💆💇🙅-🙇🙋🙍🙎🚣🚴🚵🤦🤵🤷-🤹🤽🤾🦸🦹🧍🧏🧔🧖-🧝](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🏳(?:️‍(?:⚧️|🌈))?|🏴(?:‍☠️|󠁧(?:󠁢(?:󠁥󠁮󠁧|󠁳󠁣󠁴)󠁿)?)?|🐈(?:‍⬛)?|🐕(?:‍🦺)?|🐦(?:‍[⬛🔥])?|🐻(?:‍❄️)?|👁(?:️‍🗨️)?|👨(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨|👦(?:‍👦)?|👧(?:‍[👦👧])?|[👨👩]‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|👩(?:‍(?:[⚕⚖✈]️|❤️‍(?:[👨👩]|💋‍[👨👩])|👦(?:‍👦)?|👧(?:‍[👦👧])?|👩‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[👯🤼🧞🧟](?:‍[♀♂]️)?|😮(?:‍💨)?|😵(?:‍💫)?|😶(?:‍🌫️)?|🙂(?:‍[↔↕]️)?|🧑(?:‍(?:[⚕⚖✈]️|🤝‍🧑|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]|(?:🧑‍)?🧒(?:‍🧒)?)|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏼-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏽-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏼🏾🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏽🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏾]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-☄☎☑☔☕☘☠☢☣☦☪☮☯☸-☺♀♂♈-♓♟♠♣♥♦♨♻♾♿⚒-⚗⚙⚛⚜⚠⚡⚧⚪⚫⚰⚱⚽⚾⛄⛅⛈⛎⛏⛑⛔⛩⛪⛰-⛵⛷⛸⛺⛽✂✅✈✉✏✒✔✖✝✡✨✳✴❄❇❌❎❓-❕❗❣➕-➗➡➰➿⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙🀄🃏🅰🅱🅾🅿🆎🆑-🆚🈁🈂🈚🈯🈲-🈺🉐🉑🌀-🌡🌤-🍃🍅-🍊🍌-🎄🎆-🎓🎖🎗🎙-🎛🎞-🏁🏅🏆🏈🏉🏍-🏰🏵🏷-🐇🐉-🐔🐖-🐥🐧-🐺🐼-👀👄👅👑-👥👪👹-👻👽-💀💄💈-💎💐💒-💩💫-📽📿-🔽🕉-🕎🕐-🕧🕯🕰🕳🕶-🕹🖇🖊-🖍🖤🖥🖨🖱🖲🖼🗂-🗄🗑-🗓🗜-🗞🗡🗣🗨🗯🗳🗺-😭😯-😴😷-🙁🙃🙄🙈-🙊🚀-🚢🚤-🚳🚷-🚿🛁-🛅🛋🛍-🛒🛕-🛗🛜-🛥🛩🛫🛬🛰🛳-🛼🟠-🟫🟰🤍🤎🤐-🤗🤠-🤥🤧-🤯🤺🤿-🥅🥇-🥶🥸-🦴🦷🦺🦼-🧌🧐🧠-🧿🩰-🩼🪀-🪈🪐-🪽🪿-🫂🫎-🫛🫠-🫨]|🫱(?:🏻(?:‍🫲[🏼-🏿])?|🏼(?:‍🫲[🏻🏽-🏿])?|🏽(?:‍🫲[🏻🏼🏾🏿])?|🏾(?:‍🫲[🏻-🏽🏿])?|🏿(?:‍🫲[🏻-🏾])?)?)+"
        text = re.sub(emojis, "", text)


        #unused
        # accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove

        # arabic_punc= re.compile(r'[\u0621-\u063A\u0641-\u064A\d+]+') # Keep only Arabic letters/do not remove numbers
        # text=' '.join(arabic_punc.findall(accents.sub('',text)))
        text = text.strip()
        return text

def clean_text_custom(text):
    if (text is None) or (text is np.nan) or (text == ""):
        return ""
    else:
        text = re.sub(r"\n", " ", text) # remove line breaks
        text = re.sub(r"\r", " ", text) # remove line breaks
    
        text = re.sub(r"\u201c", "\'", text)  # replace “ character with single quote
        text = re.sub(r"\u201d", "\'", text)  # replace ” character with single quote
        text = re.sub(r"\u2018", "\'", text)  # replace ‘ character with single quote
        text = re.sub(r"\u2019", "\'", text)  # replace ’ character with single quote
        text = re.sub(r'\"', "\'", text)      # replace double quotes with single quotes
        
        text = re.sub(r"http\S+[^\']", " ", text) # remove t.co urls: t.co shortened urls see https://help.twitter.com/en/using-x/how-to-post-a-link
        text = re.sub(r"RT @\w+:", "", text) # remove the pattern "RT @<username>:", but keep other usernames in the tweet, @ gets removed later
        
        # TODO: does this improve results? hashtags sometimes contain important information; see e.g. rumor AuRED_104 
        text = re.sub(r"[\.\,\|\?\?\/\=\#\@\_\t]", " ", text)  # remove special characters; also remves the # and @ but leaves the text of a hashtag or username

        text = re.sub(r'\s\:', " ", text)   # remove " :" pattern that sometimes gets "left behind" by the previous regexes

        # remove emojis; for info and updates as more emojis are released see: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
        emojis = r"(?:[0-9#*]️⃣|[☝✊-✍🎅🏂🏇👂👃👆-👐👦👧👫-👭👲👴-👶👸👼💃💅💏💑💪🕴🕺🖐🖕🖖🙌🙏🛀🛌🤌🤏🤘-🤟🤰-🤴🤶🥷🦵🦶🦻🧒🧓🧕🫃-🫅🫰🫲-🫸][🏻-🏿]?|⛓(?:️‍💥)?|[⛹🏋🏌🕵](?:️‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|❤(?:️‍[🔥🩹])?|🇦[🇨-🇬🇮🇱🇲🇴🇶-🇺🇼🇽🇿]|🇧[🇦🇧🇩-🇯🇱-🇴🇶-🇹🇻🇼🇾🇿]|🇨[🇦🇨🇩🇫-🇮🇰-🇵🇷🇺-🇿]|🇩[🇪🇬🇯🇰🇲🇴🇿]|🇪[🇦🇨🇪🇬🇭🇷-🇺]|🇫[🇮-🇰🇲🇴🇷]|🇬[🇦🇧🇩-🇮🇱-🇳🇵-🇺🇼🇾]|🇭[🇰🇲🇳🇷🇹🇺]|🇮[🇨-🇪🇱-🇴🇶-🇹]|🇯[🇪🇲🇴🇵]|🇰[🇪🇬-🇮🇲🇳🇵🇷🇼🇾🇿]|🇱[🇦-🇨🇮🇰🇷-🇻🇾]|🇲[🇦🇨-🇭🇰-🇿]|🇳[🇦🇨🇪-🇬🇮🇱🇴🇵🇷🇺🇿]|🇴🇲|🇵[🇦🇪-🇭🇰-🇳🇷-🇹🇼🇾]|🇶🇦|🇷[🇪🇴🇸🇺🇼]|🇸[🇦-🇪🇬-🇴🇷-🇹🇻🇽-🇿]|🇹[🇦🇨🇩🇫-🇭🇯-🇴🇷🇹🇻🇼🇿]|🇺[🇦🇬🇲🇳🇸🇾🇿]|🇻[🇦🇨🇪🇬🇮🇳🇺]|🇼[🇫🇸]|🇽🇰|🇾[🇪🇹]|🇿[🇦🇲🇼]|🍄(?:‍🟫)?|🍋(?:‍🟩)?|[🏃🚶🧎](?:‍(?:[♀♂]️(?:‍➡️)?|➡️)|[🏻-🏿](?:‍(?:[♀♂]️(?:‍➡️)?|➡️))?)?|[🏄🏊👮👰👱👳👷💁💂💆💇🙅-🙇🙋🙍🙎🚣🚴🚵🤦🤵🤷-🤹🤽🤾🦸🦹🧍🧏🧔🧖-🧝](?:‍[♀♂]️|[🏻-🏿](?:‍[♀♂]️)?)?|🏳(?:️‍(?:⚧️|🌈))?|🏴(?:‍☠️|󠁧(?:󠁢(?:󠁥󠁮󠁧|󠁳󠁣󠁴)󠁿)?)?|🐈(?:‍⬛)?|🐕(?:‍🦺)?|🐦(?:‍[⬛🔥])?|🐻(?:‍❄️)?|👁(?:️‍🗨️)?|👨(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨|👦(?:‍👦)?|👧(?:‍[👦👧])?|[👨👩]‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?👨[🏻-🏿]|🤝‍👨[🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|👩(?:‍(?:[⚕⚖✈]️|❤️‍(?:[👨👩]|💋‍[👨👩])|👦(?:‍👦)?|👧(?:‍[👦👧])?|👩‍(?:👦(?:‍👦)?|👧(?:‍[👦👧])?)|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳])|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏼-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏽-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻🏼🏾🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏽🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?[👨👩][🏻-🏿]|🤝‍[👨👩][🏻-🏾]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[👯🤼🧞🧟](?:‍[♀♂]️)?|😮(?:‍💨)?|😵(?:‍💫)?|😶(?:‍🌫️)?|🙂(?:‍[↔↕]️)?|🧑(?:‍(?:[⚕⚖✈]️|🤝‍🧑|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]|(?:🧑‍)?🧒(?:‍🧒)?)|🏻(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏼-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏼(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏽-🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏽(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻🏼🏾🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏾(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏽🏿]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?|🏿(?:‍(?:[⚕⚖✈]️|❤️‍(?:💋‍)?🧑[🏻-🏾]|🤝‍🧑[🏻-🏿]|[🦯🦼🦽](?:‍➡️)?|[🌾🍳🍼🎄🎓🎤🎨🏫🏭💻💼🔧🔬🚀🚒🦰-🦳]))?)?|[©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-☄☎☑☔☕☘☠☢☣☦☪☮☯☸-☺♀♂♈-♓♟♠♣♥♦♨♻♾♿⚒-⚗⚙⚛⚜⚠⚡⚧⚪⚫⚰⚱⚽⚾⛄⛅⛈⛎⛏⛑⛔⛩⛪⛰-⛵⛷⛸⛺⛽✂✅✈✉✏✒✔✖✝✡✨✳✴❄❇❌❎❓-❕❗❣➕-➗➡➰➿⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙🀄🃏🅰🅱🅾🅿🆎🆑-🆚🈁🈂🈚🈯🈲-🈺🉐🉑🌀-🌡🌤-🍃🍅-🍊🍌-🎄🎆-🎓🎖🎗🎙-🎛🎞-🏁🏅🏆🏈🏉🏍-🏰🏵🏷-🐇🐉-🐔🐖-🐥🐧-🐺🐼-👀👄👅👑-👥👪👹-👻👽-💀💄💈-💎💐💒-💩💫-📽📿-🔽🕉-🕎🕐-🕧🕯🕰🕳🕶-🕹🖇🖊-🖍🖤🖥🖨🖱🖲🖼🗂-🗄🗑-🗓🗜-🗞🗡🗣🗨🗯🗳🗺-😭😯-😴😷-🙁🙃🙄🙈-🙊🚀-🚢🚤-🚳🚷-🚿🛁-🛅🛋🛍-🛒🛕-🛗🛜-🛥🛩🛫🛬🛰🛳-🛼🟠-🟫🟰🤍🤎🤐-🤗🤠-🤥🤧-🤯🤺🤿-🥅🥇-🥶🥸-🦴🦷🦺🦼-🧌🧐🧠-🧿🩰-🩼🪀-🪈🪐-🪽🪿-🫂🫎-🫛🫠-🫨]|🫱(?:🏻(?:‍🫲[🏼-🏿])?|🏼(?:‍🫲[🏻🏽-🏿])?|🏽(?:‍🫲[🏻🏼🏾🏿])?|🏾(?:‍🫲[🏻-🏽🏿])?|🏿(?:‍🫲[🏻-🏾])?)?)+"
        text = re.sub(emojis, " ", text)

        text = re.sub(' +', ' ', text).strip() # replace multiple spaces with single space

        return text
    
if __name__ == "__main__":
    test_texts = [
        "RT @johndoe: This is a test.",
        "This is a test.",
        "This is a test. #hashtag",
        "This is a test. @johndoe",
        "This is a test. @johndoe #hashtag",
        "This is a test. @johndoe #hashtag @johndoe",
        "ISSUE: couldn't translate",
        "🔴 | Referees for Saturday's matches in the Premier League... ⬇️⬇️ #EFA https://t.co/IV7MVjEARN",
        "📽️ Watch the Deputy Chairman of Hamas in the Gaza Strip, Dr\n. Khalil Al-Hayya, in a special interview on Al-Aqsa satellite channel: UAE normalization came at a suspicious time when the occupation was encroaching on land and sanctities. It also constituted a shock to the Palestinian people and a rescue operation for Netanyahu from his internal crises https://t.co/9v6Ltnlic5",
        "“▪️ Sheikh Dr. Youssef Al-Qaradawi @alqaradawy: “We call on the nation to support the stationed in Al-Aqsa and the resistance men in Gaza, as they are defending the sanctity and dignity of Muslims in the face of an arrogant Zionist Zionist enemy.” #Gaza_under_the_bombing",
        '"The Minister of Health discusses with the German company Siemens the transfer of modern technology to support the health sector in Egypt https://t.co/ZZR1dCKqIj"'

    ]
    for text in test_texts:
        print(clean_text_custom(text))