# BrainChat labeling guide

Guide version: `cb_guide_2026_11_06_v1` (labels `cb_labels_2026_11_06_v1`).

This guide is what the labeler reads while labeling in Label Studio. Every `Definition`
line below is copied word for word into `config/labels_v1.json` `help`, and Label
Studio shows the same text as the hint on each choice; a test keeps the two equal.
Any change to a value or a definition bumps the guide version.

All examples are made up. Speakers are `s1`, `s2`, ... (a speaker number holds only
inside one window), and emotes show as made-up IDs such as `fake-laugh`.

## How to label a window

- Label the window as a whole: the shared reaction across its lines, not one message.
- Answer each of the four questions with values **or** mark it not sure with reasons,
  never both. Label Studio accepts both, but sync refuses that annotation.
- Leave tone blank when no tone stands out; that stores `no clear tone`.
- Leave the cause empty unless evidence outside chat shows it.
- The counts line under the chat (messages, posters, spread, top emote ID, chant,
  replies, questions, joining) is there to help; it is never labeled.
- Skip (cancel) a task only when it can't be labeled at all; a skipped task stores
  nothing.

## What chat is doing

Definition (`doing`): What chat as a whole is doing in this window; pick every reaction a clear share of the lines shows (at least one).

Pick every value a clear share of the lines shows.

### laughing

Definition (`doing.laughing`): Chat finds something funny: laughter words, laughing emotes, joke replies.

- `s1: LMAO` / `s2: fake-laugh fake-laugh` / `s3: im crying`
- `s1: that fall was so dumb` / `s2: HAHAHA` / `s3: fake-laugh`
- `s1: clip it` / `s2: the voice crack fake-laugh` / `s3: hahaha`

### cheering

Definition (`doing.cheering`): Chat celebrates or hypes a success: "lets go", "gg", clapping or hype emotes.

- `s1: LETS GOOO` / `s2: fake-hype fake-hype` / `s3: gg`
- `s1: first try!!` / `s2: CLEAN` / `s3: fake-clap`
- `s1: W` / `s2: W streamer` / `s3: huge W`

### shocked

Definition (`doing.shocked`): Chat reacts to something sudden or surprising: "what", "no way", "omg".

- `s1: WHAT` / `s2: no way` / `s3: omg`
- `s1: did that just happen` / `s2: fake-gasp` / `s3: ???`
- `s1: HUH` / `s2: wait what` / `s3: no shot`

### tense

Definition (`doing.tense`): Chat is on edge about what happens next: "oh no", "careful", waiting for a result.

- `s1: oh no oh no` / `s2: careful` / `s3: one hp`
- `s1: dont miss` / `s2: fake-sweat` / `s3: im shaking`
- `s1: come on come on` / `s2: please` / `s3: this is it`

### sad

Definition (`doing.sad`): Chat shows sadness, sympathy or disappointment: "rip", "nooo", crying emotes.

- `s1: rip` / `s2: nooo` / `s3: fake-cry`
- `s1: so close` / `s2: unlucky` / `s3: sad`
- `s1: F` / `s2: F` / `s3: rest in peace little guy`

### asking

Definition (`doing.asking`): Chat asks questions about what is happening, the stream or the game.

- `s1: what game is this?` / `s2: how long is the run`
- `s1: why did he reset?` / `s2: whats the bet`
- `s1: is this a new patch?` / `s2: when is the next stream?`

### advising (backseating)

Definition (`doing.advising`): Chat tells the streamer what to do: tips, instructions, backseating.

- `s1: go left` / `s2: use the shield` / `s3: heal first`
- `s1: you should craft the bow` / `s2: check the map`
- `s1: dont open that door` / `s2: jump now` / `s3: reload`

### complaining

Definition (`doing.complaining`): Chat complains or vents frustration about the stream, the game, the tech or anything else.

- `s1: this game is so broken` / `s2: worst patch ever`
- `s1: audio is bad again` / `s2: why is it so loud`
- `s1: boring` / `s2: skip this part` / `s3: ugh`

### chatting as usual

Definition (`doing.chatting_as_usual`): Ordinary chat with no shared reaction: greetings, side talk, unrelated lines.

- `s1: hi chat` / `s2: whats for dinner` / `s3: back from work`
- `s1: anyone watch the match yesterday` / `s2: yeah it was fine`
- `s1: good morning` / `s2: lurking` / `s3: hello from the made-up city`

### other

Definition (`doing.other`): A clear shared reaction that none of the other values describes.

- `s1: happy birthday!!` / `s2: hbd` / `s3: fake-cake`
- Chat counts down together for a made-up event: `s1: 3` / `s2: 2` / `s3: 1`
- Chat welcomes a raid: `s1: welcome raiders` / `s2: hi raiders`

## Aimed at

Definition (`aimed_at`): Who or what most of the reaction is about; pick one.

### streamer

Definition (`aimed_at.streamer`): The streamer: what they did, said or decided.

- `s1: you missed it lol` / `s2: why did you do that`
- `s1: nice play streamer` / `s2: you are cracked`
- `s1: your hair today lol` / `s2: love the new mic voice`

### game/content

Definition (`aimed_at.game_content`): The game or content on screen: an event, a character, a play, a scene.

- `s1: that boss is unfair` / `s2: the ending scene!!`
- `s1: this level design is wild` / `s2: the music here`
- `s1: that npc lmao` / `s2: the dragon just flew away`

### guest

Definition (`aimed_at.guest`): A guest, co-streamer or other person on stream who is not the streamer.

- `s1: the guest is so funny` / `s2: hi to the co-op partner`
- `s1: the other player carried` / `s2: guest W`
- `s1: who is the voice on call` / `s2: let the guest talk`

### chat itself

Definition (`aimed_at.chat_itself`): Chat itself: other chatters, a running chat joke, a chant, chat's own behaviour.

- `s1: chat is wild today` / `s2: chat calm down`
- `s1: fake-chant fake-chant` (a running chat joke) / `s2: we do this every stream`
- `s1: @s2 no you` / `s2: @s1 lol`

### tech problem

Definition (`aimed_at.tech_problem`): A technical problem: lag, a dropped stream, broken audio or video.

- `s1: lag` / `s2: frozen?` / `s3: buffering`
- `s1: no audio` / `s2: we cant hear you`
- `s1: stream dropped` / `s2: its back` / `s3: 360p again`

### unclear

Definition (`aimed_at.unclear`): No single target stands out: the lines point at different things or at nothing in particular.

- `s1: lol` / `s2: nice` / `s3: hi` (no shared target)
- `s1: the boss lol` / `s2: chat is weird` / `s3: lag?`
- `s1: yeah` / `s2: true` / `s3: ok`

## How chat feels about it

Definition (`feeling`): How chat feels about what it is reacting to; pick one.

### supportive

Definition (`feeling.supportive`): Chat is on the target's side: praise, encouragement, sympathy, hype.

- `s1: you got this` / `s2: so good` / `s3: proud of you`
- `s1: unlucky, next one` / `s2: still great`
- `s1: GG well played` / `s2: fake-heart`

### critical

Definition (`feeling.critical`): Chat is against the target: criticism, mockery, blame, frustration.

- `s1: thats so bad` / `s2: how did you miss that`
- `s1: throw` / `s2: classic choke`
- `s1: worst strat` / `s2: uninstall lol`

### mixed

Definition (`feeling.mixed`): Chat is split: clear support and clear criticism in the same window.

- `s1: great play` / `s2: that was lucky, not skill`
- `s1: love it` / `s2: hate it` / `s3: its fine`
- `s1: good call` / `s2: bad call, should have waited`

### neutral

Definition (`feeling.neutral`): No clear stance for or against: ordinary chatter, plain questions, plain facts.

- `s1: what level is this` / `s2: level 4`
- `s1: hi` / `s2: hello`
- `s1: the run started` / `s2: ok`

## Tone

Definition (`tone`): How chat says it; optional, and leaving it blank means no clear tone.

### playful

Definition (`tone.playful`): Joking, teasing or light-hearted, even when the words sound harsh.

- `s1: you are so bad lol` / `s2: fake-laugh` (teasing among friends)
- `s1: streamer forgot how to walk` / `s2: lmao`
- `s1: certified gamer moment` / `s2: fake-laugh`

### serious

Definition (`tone.serious`): Earnest and literal: real concern, real advice, plain statements.

- `s1: you should take a break` / `s2: hydrate`
- `s1: the setting is under audio` / `s2: try restarting`
- `s1: that was a real milestone` / `s2: congrats`

### sarcastic

Definition (`tone.sarcastic`): Says the opposite of what it means, usually to mock or tease.

- `s1: wow great aim` (after a miss) / `s2: so skilled`
- `s1: amazing stream quality` (during lag)
- `s1: totally planned` / `s2: big brain`

### hostile

Definition (`tone.hostile`): Aggressive or abusive toward someone, past playful teasing.

- `s1: get off the game` / `s2: you are garbage` (aggressive, not teasing)
- `s1: everyone report him` / `s2: leave`
- `s1: nobody likes you` / `s2: quit streaming`

### no clear tone

Definition (`tone.no_clear_tone`): No tone stands out; a real answer, not missing data.

- `s1: hi` / `s2: hello`
- `s1: what game` / `s2: the made-up one`
- `s1: ok` / `s2: brb`

## Not sure

Definition (`not_sure`): Instead of answering a question, mark it not sure with one or more reasons; a not-sure answer carries no value.

Rules:

- A not-sure answer carries **no value**, only its reasons. Pick one or more.
- Mark not sure per question: a window can have a value for what chat is doing and be
  not sure about how chat feels.
- Never guess to avoid not sure; a window marked not sure is left out of that
  question's accuracy and reported separately.
- A blank tone is not "not sure": it means no clear tone.

### unclear

Definition (`not_sure.unclear`): The lines don't let you answer: too few, too vague, or the context is off screen.

- Two lines, both `s1: ok` / `s2: ?`
- `s1: that` / `s2: yeah that` with nothing to show what "that" is
- The lines react to something only the video shows

### conflicting

Definition (`not_sure.conflicting`): The lines point to different answers and none clearly wins.

- Half the lines cheer and half complain, with no side ahead
- `s1: hype` / `s2: boring` / `s3: hype` / `s4: boring`
- Some lines aim at the streamer and as many at the guest

### cause missing

Definition (`not_sure.cause_missing`): The answer depends on what happened on stream, and chat alone doesn't show it.

- `s1: WHAT` / `s2: no way` and nothing says what happened
- `s1: rip` / `s2: rip` after an event off screen for the reader
- `s1: did he just` / `s2: he did`

### unknown in-joke

Definition (`not_sure.unknown_in_joke`): Chat is running a joke, meme or reference you don't know.

- `s1: fake-chant fake-chant` repeated with a meaning you don't know
- `s1: the made-up tuesday rule` / `s2: lol every week`
- `s1: say the line` / `s2: say it` / `s3: fake-wave`

### needs review

Definition (`not_sure.needs_review`): A contested reading, such as a contested emote ID, that needs a careful second look; never guess.

- An emote whose ID is on the contested list below
- A reading that would touch a protected trait
- A line that might be hostile to a group, where you can't tell joke from abuse

## Contested emotes

- Judge an emote only by its **exact emote ID**, never by its name. Two emotes can
  share a name and mean different things, so a name never makes a window toxic.
- Never encode an emote name as toxic or rank emotes by how funny they are.
- When a window turns on an emote ID with a contested reading (for example a
  racialized one), mark the affected question not sure with **needs review** and do
  not pick a value.
- Abstain (needs review) whenever an answer would rest on a protected trait.
- The contested emote ID list is empty in v1 (P0 uses made-up chat only). Adding an
  ID bumps the guide version, and the model's `needs review` rule fires on the IDs
  in this list.

## How it spreads (counted, never labeled)

Definition (`spread`): How the reaction moves through chat; counted from the window, shown to help, never labeled.

### repeated emotes

Definition (`spread.repeated_emotes`): Many lines repeat the same emote.

- `s1: fake-laugh` / `s2: fake-laugh` / `s3: fake-laugh fake-laugh`
- `s1: fake-hype` / `s2: gg fake-hype` / `s3: fake-hype`
- `s1: fake-cry` / `s2: fake-cry` / `s3: rip fake-cry`

### chant

Definition (`spread.chant`): Many lines repeat the same word or phrase.

- `s1: one more game` / `s2: one more game` / `s3: one more game`
- `s1: run it back` / `s2: run it back`
- `s1: made up chant` / `s2: made up chant` / `s3: made up chant`

### back-and-forth

Definition (`spread.back_and_forth`): Chatters reply to each other in a conversation.

- `s1: @s2 no way` / `s2: @s1 yes way`
- `s1: @s3 which build` / `s3: @s1 the fast one`
- `s2: @s4 true` / `s4: @s2 thanks`

### separate messages

Definition (`spread.separate_messages`): Many posters write their own lines with no shared pattern.

- `s1: hi` / `s2: nice` / `s3: what game` / `s4: lol` / `s5: gg`
- Many posters, each with a different line
- `s1: brb` / `s2: hello` / `s3: wow` / `s4: hm` / `s5: ok`

### few posters

Definition (`spread.few_posters`): Too few distinct posters to call it a chat-wide reaction.

- Only `s1` and `s2` post in the window
- `s1` writes ten lines and two others write one each
- Three posters in a two-minute window

## Cause

Definition (`cause`): Optional note on what caused the reaction, filled only from evidence outside chat; empty by default.

- Fill it only from the clip, the game's own state or the stream title; never from
  chat alone (the model says only what chat shows).
- Write a short plain note with no usernames and no copied chat lines.
- A note needs a source, and a source needs a note.

### clip

Definition (`cause_source.clip`): The cause note comes from watching the clip.

- The made-up clip shows the streamer falling off a ledge
- The made-up clip shows a boss dying on the first try
- The made-up clip shows the stream freezing

### game

Definition (`cause_source.game`): The cause note comes from the game's own state or events.

- The made-up game's scoreboard shows the round was lost
- The made-up game's log shows a rare item drop
- The made-up game's state shows the final level started

### stream title

Definition (`cause_source.stream_title`): The cause note comes from the stream title.

- The made-up title says "first attempt at the final boss"
- The made-up title says "birthday stream"
- The made-up title says "new patch day"

## Forbidden outputs

Labels, notes and anything built from them never hold:

- raw chat (no copied lines in a cause note);
- usernames or display names, or anything that ties a line to a chatter;
- public toxicity scores, for a chatter, a channel or an emote;
- claims about a chatter's identity, including protected traits.
