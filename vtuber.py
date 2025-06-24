import pygame
import numpy as np
import os
import tempfile

# --- NEW: LangChain & Google Generative AI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- CONFIGURATION ---
GOOGLE_API_KEY = "AIzaSyCnIp1gOO0yu2w6WDgVr4WySAH-a-Icsz4"  # Replace with your Google API key

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Setup LangChain Google GenAI
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-2.0-flash"
)

# Setup pygame
pygame.init()
screen = pygame.display.set_mode((600, 700))
pygame.display.set_caption("VTuber Sprite")
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 20)

# Load sprites (update these paths to your local images)
try:
    happy_sprite = pygame.image.load(r"C:\Users\Krishna Cheliaah\Downloads\download.jpg")
    sad_sprite = pygame.image.load(r"C:\Users\Krishna Cheliaah\Downloads\Sad-face.png")
    neutral_sprite = pygame.image.load(r"C:\Users\Krishna Cheliaah\Downloads\images.jpg")
except pygame.error as e:
    print(f"Error loading sprites: {e}")
    # Create simple colored rectangles as fallback sprites
    happy_sprite = pygame.Surface((200, 200))
    happy_sprite.fill((255, 255, 0))  # Yellow for happy
    sad_sprite = pygame.Surface((200, 200))
    sad_sprite.fill((0, 0, 255))      # Blue for sad
    neutral_sprite = pygame.Surface((200, 200))
    neutral_sprite.fill((128, 128, 128))  # Gray for neutral

current_sprite = neutral_sprite

# UI elements
input_box = pygame.Rect(50, 600, 400, 40)
ask_button = pygame.Rect(470, 600, 80, 40)
quit_button = pygame.Rect(470, 650, 80, 40)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
input_text = ""
output_text = ""

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.5:
        return 'happy'
    elif score <= -0.5:
        return 'sad'
    else:
        return 'neutral'

def speak_with_edge_tts(text, filename="output.wav"):
    """TTS using Edge TTS (free Microsoft TTS)"""
    try:
        import edge_tts
        import asyncio
        
        async def generate_speech():
            # Use a nice female voice - you can change this to other voices like:
            # "en-US-AriaNeural", "en-US-SaraNeural", "en-US-ZiraNeural"
            communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
            await communicate.save(filename)
        
        asyncio.run(generate_speech())
        
        # Convert to pygame-compatible format using pydub
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            # Load the audio file and convert to pygame-compatible format
            audio = AudioSegment.from_file(filename)
            # Convert to 16-bit PCM WAV format that pygame likes
            audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
            
            # Save the converted file
            converted_filename = filename.replace('.wav', '_converted.wav')
            audio.export(converted_filename, format="wav")
            return converted_filename
            
        except ImportError:
            print("pydub not available, trying direct playback...")
            return filename
        
    except ImportError:
        raise Exception("edge-tts not installed. Please run: pip install edge-tts")
    except Exception as e:
        raise Exception(f"Edge TTS failed: {e}")

def generate_speech(text):
    """Generate speech using Edge TTS"""
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, "vtuber_output.wav")
    
    try:
        print("Generating speech with Edge TTS...")
        return speak_with_edge_tts(text, filename)
    except Exception as e:
        print(f"Edge TTS failed: {e}")
        return None

def vtuber_respond(user_text):
    """Use LangChain Google GenAI to get the answer"""
    try:
        response = llm.invoke(user_text)
        answer = response.content
        # Limit response length for TTS
        if len(answer) > 500:
            answer = answer[:500] + "..."
        return answer
    except Exception as e:
        print(f"AI response error: {e}")
        return "Sorry, I couldn't get a response right now."

def respond(user_text):
    global current_sprite, output_text
    
    print(f"User input: {user_text}")
    
    # First, analyze user's sentiment and update sprite immediately
    user_mood = get_sentiment(user_text)
    print(f"User sentiment: {user_mood}")
    current_sprite = {
        "happy": happy_sprite,
        "sad": sad_sprite,
        "neutral": neutral_sprite
    }.get(user_mood, neutral_sprite)
    
    # Get AI response
    answer = vtuber_respond(user_text)
    output_text = answer[:100] + "..." if len(answer) > 100 else answer  # Limit display text
    
    # Then analyze AI response sentiment and update sprite again if different
    ai_mood = get_sentiment(answer)
    print(f"AI response sentiment: {ai_mood}")
    
    # Use AI response sentiment to determine final sprite
    # This creates a more natural conversation flow where the VTuber reacts to the user
    # but then settles into their own emotional state based on their response
    final_sprite = {
        "happy": happy_sprite,
        "sad": sad_sprite,
        "neutral": neutral_sprite
    }.get(ai_mood, neutral_sprite)
    
    # If sentiments are different, show user sentiment briefly, then AI sentiment
    if user_mood != ai_mood:
        print(f"Sentiment changed from {user_mood} (user) to {ai_mood} (AI response)")
    
    current_sprite = final_sprite
    
    # Generate and play speech
    try:
        wav_file = generate_speech(answer)
        if wav_file and os.path.exists(wav_file):
            # Initialize pygame mixer with specific settings for better compatibility
            pygame.mixer.quit()  # Stop any existing mixer
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=1024)
            pygame.mixer.init()
            
            try:
                # Try loading and playing the audio
                pygame.mixer.music.load(wav_file)
                pygame.mixer.music.play()
                print(f"Playing audio from: {wav_file}")
            except pygame.error as e:
                print(f"Pygame audio error: {e}")
                # Fallback: try using pygame.mixer.Sound instead
                try:
                    sound = pygame.mixer.Sound(wav_file)
                    sound.play()
                    print("Played using pygame.mixer.Sound")
                except pygame.error as e2:
                    print(f"Sound playback also failed: {e2}")
                    
        else:
            print("No audio file generated")
    except Exception as e:
        print(f"Audio playback error: {e}")

# Main game loop
running = True
clock = pygame.time.Clock()

print("VTuber started! Type your message and press Enter or click Ask.")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if input_box.collidepoint(event.pos):
                active = True
                color = color_active
            else:
                active = False
                color = color_inactive
            if ask_button.collidepoint(event.pos):
                if input_text.strip():
                    respond(input_text)
                    input_text = ""
            if quit_button.collidepoint(event.pos):
                running = False

        elif event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    if input_text.strip():
                        respond(input_text)
                        input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode

    # Clear screen
    screen.fill((255, 255, 255))
    
    # Draw sprite
    sprite_rect = current_sprite.get_rect()
    sprite_rect.center = (300, 250)
    screen.blit(current_sprite, sprite_rect)

    # Draw input box
    pygame.draw.rect(screen, color, input_box, 2)
    txt_surface = font.render(input_text, True, (0, 0, 0))
    screen.blit(txt_surface, (input_box.x+5, input_box.y+5))

    # Draw output box with word wrapping
    output_box = pygame.Rect(50, 450, 500, 120)
    pygame.draw.rect(screen, pygame.Color('gray90'), output_box)
    pygame.draw.rect(screen, pygame.Color('black'), output_box, 2)
    
    # Word wrap the output text
    words = output_text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if small_font.size(test_line)[0] < output_box.width - 10:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    
    # Display lines
    y_offset = 5
    for i, line in enumerate(lines[:5]):  # Max 5 lines
        if i == 0:
            line = "VTuber: " + line
        output_surface = small_font.render(line, True, (0, 0, 0))
        screen.blit(output_surface, (output_box.x + 5, output_box.y + y_offset))
        y_offset += 20

    # Draw buttons
    pygame.draw.rect(screen, pygame.Color('lightgreen'), ask_button)
    ask_text = small_font.render("Ask", True, (0, 0, 0))
    screen.blit(ask_text, (ask_button.x+20, ask_button.y+10))

    pygame.draw.rect(screen, pygame.Color('salmon'), quit_button)
    quit_text = small_font.render("Quit", True, (0, 0, 0))
    screen.blit(quit_text, (quit_button.x+15, quit_button.y+10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("VTuber closed!")
