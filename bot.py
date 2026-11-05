"""
Versatility bot
- Server stats (slash command + auto-updating channel name)
- NQN-style emojis for users without Nitro (webhook replacement)
- Ticket system (button panel -> private ticket channel)
- Modmail (DMs forwarded to a staff channel, staff replies sent back)
- AI chat powered by NVIDIA's OpenAI-compatible API
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict, deque

import discord
from aiohttp import web
from discord import app_commands
from discord.ext import commands, tasks
from discord.utils import MISSING
from openai import AsyncOpenAI

# ---------------------------------------------------------------- config
TOKEN = os.environ["DISCORD_TOKEN"]
NVIDIA_KEY = os.environ["NVIDIA_API_KEY"]
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
PORT = int(os.getenv("PORT", "10000"))  # Render sets PORT automatically
DATA_FILE = os.getenv("DATA_FILE", "data.json")
SYSTEM_PROMPT = os.getenv(
    "AI_SYSTEM_PROMPT",
    "You are a friendly, helpful assistant inside a Discord server. "
    "Keep answers concise and well formatted.",
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("versatility")


# ---------------------------------------------------------------- storage
def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"guilds": {}, "modmail": {}}


def save_data():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


data = load_data()


def gcfg(guild_id: int) -> dict:
    return data["guilds"].setdefault(str(guild_id), {})


def modmail_user_for(channel_id: int):
    return next((u for u, c in data["modmail"].items() if c == channel_id), None)


# ---------------------------------------------------------------- bot
intents = discord.Intents.default()
intents.message_content = True
intents.members = True


class VersatilityBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)
        self.ai = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_KEY)
        self.history = defaultdict(lambda: deque(maxlen=12))

    async def setup_hook(self):
        # Persistent views so buttons keep working after a restart
        self.add_view(TicketPanelView())
        self.add_view(TicketCloseView())
        await start_web()
        update_stat_channels.start()
        await self.tree.sync()
        log.info("Slash commands synced")


bot = VersatilityBot()


async def start_web():
    """Tiny HTTP server so Render sees an open port and UptimeRobot can ping it."""
    app = web.Application()
    app.router.add_get("/", lambda request: web.Response(text="Bot is alive"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PORT).start()


@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id)


# ---------------------------------------------------------------- server stats
def stats_values(guild: discord.Guild) -> dict:
    bots_count = sum(1 for m in guild.members if m.bot)
    return {
        "members": guild.member_count,
        "humans": guild.member_count - bots_count,
        "bots": bots_count,
        "boosts": guild.premium_subscription_count or 0,
        "channels": len(guild.channels),
        "roles": len(guild.roles),
    }


@tasks.loop(minutes=10)  # Discord limits channel renames, so keep this slow
async def update_stat_channels():
    for gid, cfg in list(data["guilds"].items()):
        channel_id = cfg.get("stats_channel")
        guild = bot.get_guild(int(gid))
        if not channel_id or guild is None:
            continue
        channel = guild.get_channel(channel_id)
        if channel is None:
            continue
        template = cfg.get("stats_template", "👥 Members: {members}")
        try:
            name = template.format(**stats_values(guild))[:100]
        except (KeyError, ValueError):
            name = "👥 Members: {}".format(guild.member_count)
        if channel.name != name:
            try:
                await channel.edit(name=name)
            except discord.HTTPException as e:
                log.warning("Could not rename stats channel: %s", e)


@bot.tree.command(name="stats", description="Show server statistics")
async def stats(interaction: discord.Interaction):
    g = interaction.guild
    v = stats_values(g)
    embed = discord.Embed(title=f"📊 {g.name}", color=0x5865F2)
    if g.icon:
        embed.set_thumbnail(url=g.icon.url)
    embed.add_field(name="Members", value=v["members"])
    embed.add_field(name="Humans", value=v["humans"])
    embed.add_field(name="Bots", value=v["bots"])
    embed.add_field(name="Channels", value=v["channels"])
    embed.add_field(name="Roles", value=v["roles"])
    embed.add_field(name="Boosts", value=f"{v['boosts']} (Tier {g.premium_tier})")
    embed.add_field(name="Created", value=discord.utils.format_dt(g.created_at, "D"))
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="stats_channel", description="Make a channel show live stats in its name")
@app_commands.default_permissions(manage_channels=True)
@app_commands.describe(
    channel="Channel to rename automatically",
    template="Name template. Use {members}, {humans}, {bots}, {boosts}, {channels}, {roles}",
)
async def stats_channel(
    interaction: discord.Interaction,
    channel: discord.abc.GuildChannel,
    template: str = "👥 Members: {members}",
):
    cfg = gcfg(interaction.guild_id)
    cfg["stats_channel"] = channel.id
    cfg["stats_template"] = template
    save_data()
    await interaction.response.send_message(
        f"Done. {channel.mention} will update about every 10 minutes.", ephemeral=True
    )


# ---------------------------------------------------------------- NQN (emoji for non-Nitro)
EMOJI_RE = re.compile(r":([A-Za-z0-9_]{2,32}):")


async def get_webhook(channel: discord.TextChannel) -> discord.Webhook:
    for wh in await channel.webhooks():
        if wh.user and wh.user.id == bot.user.id:
            return wh
    return await channel.create_webhook(name="Versatility NQN")


async def handle_nqn(message: discord.Message) -> bool:
    cfg = gcfg(message.guild.id)
    if not cfg.get("nqn"):
        return False
    me = message.guild.me
    if not (me.guild_permissions.manage_webhooks and me.guild_permissions.manage_messages):
        return False

    own_names = {e.name for e in message.guild.emojis}
    changed = False

    def swap(match: re.Match) -> str:
        nonlocal changed
        name = match.group(1)
        if name in own_names:  # this server already has it, no need to swap
            return match.group(0)
        emoji = discord.utils.get(bot.emojis, name=name)
        if emoji is None:
            return match.group(0)
        changed = True
        return str(emoji)  # gives <:name:id> or <a:name:id>

    new_content = EMOJI_RE.sub(swap, message.content)
    if not changed:
        return False

    channel = message.channel
    thread = None
    if isinstance(channel, discord.Thread):
        thread = channel
        channel = channel.parent

    webhook = await get_webhook(channel)
    files = [await a.to_file() for a in message.attachments]
    await webhook.send(
        content=new_content,
        username=message.author.display_name[:80],
        avatar_url=message.author.display_avatar.url,
        files=files,
        thread=thread or MISSING,
        allowed_mentions=discord.AllowedMentions.none(),
    )
    try:
        await message.delete()
    except discord.HTTPException:
        pass
    return True


@bot.tree.command(name="nqn", description="Let non-Nitro users use emojis from servers I'm in")
@app_commands.default_permissions(manage_guild=True)
async def nqn(interaction: discord.Interaction, enabled: bool):
    gcfg(interaction.guild_id)["nqn"] = enabled
    save_data()
    state = "enabled" if enabled else "disabled"
    await interaction.response.send_message(f"NQN is now {state}.", ephemeral=True)


# ---------------------------------------------------------------- tickets
class TicketCloseView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Close ticket", emoji="🔒", style=discord.ButtonStyle.danger,
                       custom_id="vb:ticket_close")
    async def close(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Closing this ticket in 5 seconds...")
        await asyncio.sleep(5)
        await interaction.channel.delete(reason=f"Ticket closed by {interaction.user}")


class TicketPanelView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Open a ticket", emoji="🎫", style=discord.ButtonStyle.primary,
                       custom_id="vb:ticket_open")
    async def open_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        guild = interaction.guild
        cfg = gcfg(guild.id)
        category = guild.get_channel(cfg.get("ticket_category", 0))
        if not isinstance(category, discord.CategoryChannel):
            return await interaction.response.send_message(
                "Tickets are not set up yet. An admin must run /ticket_setup.", ephemeral=True
            )

        # The user's ID is stored in the channel topic, so we can find an open ticket
        uid = str(interaction.user.id)
        existing = [c for c in category.text_channels if c.topic == uid]
        if existing:
            return await interaction.response.send_message(
                f"You already have a ticket: {existing[0].mention}", ephemeral=True
            )

        overwrites = {
            guild.default_role: discord.PermissionOverwrite(view_channel=False),
            interaction.user: discord.PermissionOverwrite(
                view_channel=True, send_messages=True, read_message_history=True, attach_files=True
            ),
            guild.me: discord.PermissionOverwrite(
                view_channel=True, send_messages=True, manage_channels=True
            ),
        }
        for rid in cfg.get("support_roles", []):
            role = guild.get_role(rid)
            if role:
                overwrites[role] = discord.PermissionOverwrite(view_channel=True, send_messages=True)

        channel = await guild.create_text_channel(
            f"ticket-{interaction.user.name}"[:95],
            category=category,
            overwrites=overwrites,
            topic=uid,
        )
        await channel.send(
            f"{interaction.user.mention} Thanks for opening a ticket! "
            "Describe your issue and the staff will reply soon.",
            view=TicketCloseView(),
        )
        await interaction.response.send_message(f"Ticket created: {channel.mention}", ephemeral=True)


@bot.tree.command(name="ticket_setup", description="Set where tickets are created and who can see them")
@app_commands.default_permissions(manage_guild=True)
@app_commands.describe(category="Category where ticket channels are created",
                       support_role="Role that can see and answer tickets")
async def ticket_setup(interaction: discord.Interaction, category: discord.CategoryChannel,
                       support_role: discord.Role | None = None):
    cfg = gcfg(interaction.guild_id)
    cfg["ticket_category"] = category.id
    if support_role:
        roles = cfg.setdefault("support_roles", [])
        if support_role.id not in roles:
            roles.append(support_role.id)
    save_data()
    await interaction.response.send_message("Ticket setup saved. Now run /ticket_panel.", ephemeral=True)


@bot.tree.command(name="ticket_panel", description="Post the ticket button in this channel")
@app_commands.default_permissions(manage_guild=True)
async def ticket_panel(interaction: discord.Interaction):
    embed = discord.Embed(
        title="Support tickets",
        description="Click the button below to open a private ticket with the staff team.",
        color=0x5865F2,
    )
    await interaction.channel.send(embed=embed, view=TicketPanelView())
    await interaction.response.send_message("Panel posted.", ephemeral=True)


# ---------------------------------------------------------------- modmail
@bot.tree.command(name="modmail_setup", description="Set the category where modmail threads are created")
@app_commands.default_permissions(manage_guild=True)
async def modmail_setup(interaction: discord.Interaction, category: discord.CategoryChannel):
    gcfg(interaction.guild_id)["modmail_category"] = category.id
    save_data()
    await interaction.response.send_message(
        "Modmail is set. Users can now DM me to reach your staff.", ephemeral=True
    )


@bot.tree.command(name="modmail_close", description="Close the modmail thread in this channel")
@app_commands.default_permissions(manage_messages=True)
async def modmail_close(interaction: discord.Interaction):
    uid = modmail_user_for(interaction.channel_id)
    if uid is None:
        return await interaction.response.send_message("Use this inside a modmail channel.", ephemeral=True)
    del data["modmail"][uid]
    save_data()
    await interaction.response.send_message("Closing this modmail thread...")
    try:
        user = await bot.fetch_user(int(uid))
        await user.send("This modmail thread was closed by staff. DM me again if you need more help.")
    except discord.HTTPException:
        pass
    await interaction.channel.delete(reason="Modmail closed")


async def handle_dm(message: discord.Message):
    """A user DM'd the bot: forward it to the modmail channel of a shared server."""
    target_guild, target_cat = None, None
    for gid, cfg in data["guilds"].items():
        guild = bot.get_guild(int(gid))
        cat_id = cfg.get("modmail_category")
        if guild and cat_id and guild.get_member(message.author.id):
            target_guild, target_cat = guild, guild.get_channel(cat_id)
            break
    if not isinstance(target_cat, discord.CategoryChannel):
        return await message.channel.send("Modmail is not set up on a server we share.")

    uid = str(message.author.id)
    channel = target_guild.get_channel(data["modmail"].get(uid, 0))
    if channel is None:
        channel = await target_guild.create_text_channel(
            f"modmail-{message.author.name}"[:95], category=target_cat, topic=uid
        )
        data["modmail"][uid] = channel.id
        save_data()
        await channel.send(embed=discord.Embed(
            title="New modmail thread",
            description=f"{message.author} (`{uid}`)\nReply here to answer. Use `//` at the start for internal notes.",
            color=0xFEE75C,
        ))

    embed = discord.Embed(description=message.content or "*(no text)*", color=0x57F287)
    embed.set_author(name=str(message.author), icon_url=message.author.display_avatar.url)
    await channel.send(embed=embed, files=[await a.to_file() for a in message.attachments])
    await message.add_reaction("✅")


async def handle_staff_reply(message: discord.Message) -> bool:
    uid = modmail_user_for(message.channel.id)
    if uid is None:
        return False
    if message.content.startswith("//"):  # internal note, not sent to the user
        return True
    try:
        user = await bot.fetch_user(int(uid))
        embed = discord.Embed(description=message.content or "*(attachment)*", color=0x5865F2)
        embed.set_author(name=f"{message.author.display_name} (Staff)")
        await user.send(embed=embed, files=[await a.to_file() for a in message.attachments])
        await message.add_reaction("📨")
    except discord.HTTPException:
        await message.channel.send("Could not DM this user. They may have DMs closed.")
    return True


# ---------------------------------------------------------------- AI (NVIDIA)
def chunks(text: str, size: int = 1900):
    return [text[i:i + size] for i in range(0, len(text), size)] or ["(empty response)"]


async def ask_ai(channel_id: int, user_text: str) -> str:
    history = bot.history[channel_id]
    history.append({"role": "user", "content": user_text[:4000]})
    response = await bot.ai.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        max_tokens=800,
        temperature=0.7,
    )
    answer = response.choices[0].message.content or ""
    history.append({"role": "assistant", "content": answer})
    return answer


@bot.tree.command(name="ask", description="Ask the AI anything")
@app_commands.describe(prompt="What do you want to ask?")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = await ask_ai(interaction.channel_id, prompt)
    except Exception:
        log.exception("NVIDIA API call failed")
        answer = "The AI is unavailable right now. Please try again in a moment."
    for part in chunks(answer):
        await interaction.followup.send(part)


@bot.tree.command(name="ai_reset", description="Clear the AI's memory in this channel")
async def ai_reset(interaction: discord.Interaction):
    bot.history.pop(interaction.channel_id, None)
    await interaction.response.send_message("AI memory cleared for this channel.", ephemeral=True)


async def handle_ai_trigger(message: discord.Message) -> bool:
    """Respond when the bot is @mentioned or when someone replies to one of its messages."""
    mentioned = bot.user in message.mentions
    replied = (
        message.reference is not None
        and isinstance(message.reference.resolved, discord.Message)
        and message.reference.resolved.author.id == bot.user.id
    )
    if not (mentioned or replied):
        return False
    text = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    if not text:
        return False
    async with message.channel.typing():
        try:
            answer = await ask_ai(message.channel.id, text)
        except Exception:
            log.exception("NVIDIA API call failed")
            answer = "The AI is unavailable right now. Please try again in a moment."
    for part in chunks(answer):
        await message.reply(part, mention_author=False)
    return True


# ---------------------------------------------------------------- message router
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.guild is None:
        return await handle_dm(message)
    if await handle_staff_reply(message):
        return
    if await handle_nqn(message):
        return
    await handle_ai_trigger(message)


if __name__ == "__main__":
    bot.run(TOKEN, log_handler=None)
