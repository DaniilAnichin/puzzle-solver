import logging
from tempfile import NamedTemporaryFile
from os import getenv

from telegram import ReplyKeyboardRemove, Update, ext, BotCommand

from puzzle_solver.detector import detect_n_solve, get_buffer_from_opencv_img, get_opencv_img_from_buffer

HELLO_TEMPLATE = """
Hi, {username}! You can send me screenshot from polygrams app (blocks & rectangles sections),
and I will do my best to solve the level sent.
"""
HELP_TEXT = """This bot can solve puzzles from Polygrams app (blocks & rectangles only, so far). 
To use it, send a screenshot of the level, and it will respond with the correct tile position, if able to find.
""".strip()


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def help(update: Update, context: ext.ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(HELP_TEXT)


async def start(update: Update, context: ext.ContextTypes.DEFAULT_TYPE):
    logger.info('Running `start` with user=%s', update.effective_message.from_user.username)
    # This needs to be run only once buttons change, afaik
    await context.bot.set_my_commands([
        BotCommand('/start', 'Start bot'),
        BotCommand('/help', 'Get help'),
    ])

    username = update.effective_message.from_user.full_name
    await update.effective_message.reply_text(
        HELLO_TEMPLATE.format(username=username),
        reply_markup=ReplyKeyboardRemove(),
    )


async def fallback(update: Update, context: ext.ContextTypes.DEFAULT_TYPE):
    logger.debug('Running `fallback` with user=%s', update.effective_message.from_user.username)
    logger.info('Fallback handler got: %s', update.effective_message)
    await update.effective_message.reply_text("Sorry, I did not get it.")


async def solve(update: Update, context: ext.ContextTypes.DEFAULT_TYPE):
    logger.info('Running `solve` with user=%s', update.effective_message.from_user.username)
    attachment = update.effective_message.effective_attachment

    photo = await attachment[-1].get_file()
    with NamedTemporaryFile('wb+') as out:
        await photo.download_to_memory(out)
        out.seek(0)
        img = get_opencv_img_from_buffer(out, flags=-1)

    await update.effective_message.reply_text(
        'processing..', reply_to_message_id=update.effective_message.id,
    )
    try:
        solution = detect_n_solve(img)
    except Exception as e:
        logger.exception(e)
        await update.effective_message.reply_text('Was not able to detect puzzle on image(')
        return

    if solution is not None:
        await update.effective_message.reply_text('Here you go:')
        buffer = get_buffer_from_opencv_img(solution)
        await update.effective_message.reply_photo(buffer)
    else:
        await update.effective_message.reply_text('No solution found(')


def main():
    token = getenv('BOT_TOKEN')
    if not token:
        print('No BOT_TOKEN to work with')
        exit(1)

    persistence = ext.PicklePersistence('./pkl.db')
    app = ext.ApplicationBuilder().token(token).persistence(persistence).build()

    app.add_handler(ext.CommandHandler('start', start))
    app.add_handler(ext.CommandHandler('help', help))
    app.add_handler(ext.MessageHandler(ext.filters.PHOTO, solve))
    app.add_handler(ext.MessageHandler(ext.filters.ALL, fallback))
    app.run_polling()


if __name__ == '__main__':
    main()
