#         S c r i p t   f o r   i n s t a l l i n g   t h e   r e q u i r e d   d e p e n d e n c i e d   f o r   
 #         t h e   Gadgetron and I S M R M   R a w   D a t a   F o r m a t   a n d   G a d g e t r o n   o n   W i n d o w s . 
 #         
 #         P r e r e q u i s i t e s : 
 #                 -   W i n d o w s   7   ( 6 4 - b i t ) 
 #                 -   V i s u a l   S t u d i o   ( C / C + + )   i n s t a l l e d 
 #  
 f u n c t i o n   d o w n l o a d _ f i l e ( $ u r l , $ d e s t i n a t i o n )   { 
         # L e t ' s   s e t   u p   a   w e b c l i e n t   f o r   a l l   t h e   f i l e s   w e   h a v e   t o   d o w n l o a d 
         $ c l i e n t   =   N e w - O b j e c t   S y s t e m . N e t . W e b C l i e n t 
         $ c l i e n t . D o w n l o a d F i l e ( $ u r l , $ d e s t i n a t i o n ) 
 }  
 f u n c t i o n   u n z i p ( $ z i p P a t h ,   $ d e s t i n a t i o n ) { 
         $ s h e l l   =   n e w - o b j e c t   - c o m   s h e l l . a p p l i c a t i o n ; 
         $ z i p   =   $ s h e l l . N a m e S p a c e ( $ z i p P a t h ) ; 
         i f   ( ( T e s t - P a t h   - p a t h   $ d e s t i n a t i o n )   - n e   $ T r u e ) 
         { 
                 N e w - I t e m   $ d e s t i n a t i o n   - t y p e   d i r e c t o r y 
         } 
         f o r e a c h ( $ i t e m   i n   $ z i p . i t e m s ( ) ) { 
                 $ s h e l l . N a m e s p a c e ( $ d e s t i n a t i o n ) . c o p y h e r e ( $ i t e m ) 
         } 
 }  
 f u n c t i o n   S e t - V S - E n v i r o n m e n t   ( )   { 
         $ f i l e   =   ( G e t - C h i l d I t e m   E n v : V S 1 0 0 C O M N T O O L S ) . V a l u e   +   " v s v a r s 3 2 . b a t " 
         $ c m d   =   " ` " $ f i l e ` "   &   s e t " 
         c m d   / c   $ c m d   |   F o r e a c h - O b j e c t   { 
               $ p ,   $ v   =   $ _ . s p l i t ( ' = ' ) 
               S e t - I t e m   - p a t h   e n v : $ p   - v a l u e   $ v 
       } 
 }  
 f u n c t i o n   a d d _ p a t h ( $ p a t h n a m e )   { 
         i f   ( $ e n v : p a t h     - m a t c h   [ r e g e x ] : : e s c a p e ( $ p a t h n a m e ) )   { 
                 W r i t e - H o s t   " P a t h   $ p a t h   a l r e a d y   e x i s t s " 
         }   e l s e   { 
                 s e t x   P A T H   " $ e n v : p a t h ; $ p a t h n a m e "   - m 
         } 
 }  
 W r i t e - H o s t   "Gadgetron and  I S M R M R D   R a w   D a t a   F o r m a t   D e p e n d e n c i e s   I n s t a l l a t i o n "  
 $ l i b r a r y _ l o c a t i o n   =   " C : \ M R I L i b r a r i e s " 
 $ d o w n l o a d _ l o c a t i o n   =   " C : \ M R I L i b r a r i e s \ d o w n l o a d s "  
 # L e t ' s   f i r s t   c h e c k   i f   w e   h a v e   t h e   l i b r a r y   f o l d e r   a n d   i f   n o t   c r e a t e   i t 
 i f   ( ( T e s t - P a t h   - p a t h   $ l i b r a r y _ l o c a t i o n )   - n e   $ T r u e ) 
 { 
         W r i t e - H o s t   " L i b r a r y   l o c a t i o n :   "   $ l i b r a r y _ l o c a t i o n   "   n o t   f o u n d ,   c r e a t i n g " 
         N e w - I t e m   $ l i b r a r y _ l o c a t i o n   - t y p e   d i r e c t o r y 
 } 
 e l s e 
 { 
         W r i t e - H o s t   " L i b r a r y   l o c a t i o n :   "   $ l i b r a r y _ l o c a t i o n   "   f o u n d . " 
 }  
 # N o w   c h e c k   i f   w e   h a v e   t h e   l i b r a r y   f o l d e r   a n d   i f   n o t   c r e a t e   i t 
 i f   ( ( T e s t - P a t h   - p a t h   $ d o w n l o a d _ l o c a t i o n )   - n e   $ T r u e ) 
 { 
         W r i t e - H o s t   " D o w n l o a d   l o c a t i o n :   "   $ d o w n l o a d _ l o c a t i o n   "   n o t   f o u n d ,   c r e a t i n g " 
         N e w - I t e m   $ d o w n l o a d _ l o c a t i o n   - t y p e   d i r e c t o r y 
 } 
 e l s e 
 { 
         W r i t e - H o s t   " D o w n l o a d   l o c a t i o n :   "   $ d o w n l o a d _ l o c a t i o n   "   f o u n d . " 
 }  
 # D o w n l o a d   a n d   i n s t a l l   C M A K E 
 d o w n l o a d _ f i l e   " h t t p : / / w w w . c m a k e . o r g / f i l e s / v 2 . 8 / c m a k e - 2 . 8 . 9 - w i n 3 2 - x 8 6 . e x e "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ c m a k e - 2 . 8 . 9 - w i n 3 2 - x 8 6 . e x e " ) 
 &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ c m a k e - 2 . 8 . 9 - w i n 3 2 - x 8 6 . e x e " )  
 # D o w n l o a d   a n d   i n s t a l l   G i t  d o w n l o a d _ f i l e   " h t t p : / / m s y s g i t . g o o g l e c o d e . c o m / f i l e s / G i t - 1 . 7 . 1 1 - p r e v i e w 2 0 1 2 0 7 1 0 . e x e "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ G i t - 1 . 7 . 1 1 - p r e v i e w 2 0 1 2 0 7 1 0 . e x e " )  &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ G i t - 1 . 7 . 1 1 - p r e v i e w 2 0 1 2 0 7 1 0 . e x e " )  
 # D o w n l o a d ,   u n z i p ,   a n d   i n s t a l l   H D F 5 
 d o w n l o a d _ f i l e   " h t t p : / / w w w . h d f g r o u p . o r g / f t p / H D F 5 / c u r r e n t / b i n / w i n d o w s / H D F 5 1 8 9 - w i n 6 4 - v s 1 0 - s h a r e d . z i p "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ H D F 5 1 8 9 - w i n 6 4 - v s 1 0 - s h a r e d . z i p " ) 
 u n z i p   ( $ d o w n l o a d _ l o c a t i o n   +   " \ H D F 5 1 8 9 - w i n 6 4 - v s 1 0 - s h a r e d . z i p " )     " $ d o w n l o a d _ l o c a t i o n \ h d f 5 _ b i n a r i e s " 
 &   " $ d o w n l o a d _ l o c a t i o n \ h d f 5 _ b i n a r i e s \ H D F 5 - 1 . 8 . 9 - w i n 6 4 . e x e "  
# D o w n l o a d ,   i n s t a l l   H D F V i e w 
 d o w n l o a d _ f i l e   " h t t p : / / w w w . h d f g r o u p . o r g / f t p / H D F 5 / h d f - j a v a / h d f v i e w / h d f v i e w _ i n s t a l l _ w i n 6 4 . e x e "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ h d f v i e w _ i n s t a l l _ w i n 6 4 . e x e " ) 
 &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ h d f v i e w _ i n s t a l l _ w i n 6 4 . e x e " )  
 # D o w n l o a d   a n d   i n s t a l l   C o d e S y n t h e s i s   X S D 
 d o w n l o a d _ f i l e   " h t t p : / / w w w . c o d e s y n t h e s i s . c o m / d o w n l o a d / x s d / 3 . 3 / w i n d o w s / i 6 8 6 / x s d - 3 . 3 . m s i "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ x s d - 3 . 3 . m s i " ) 
 &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ x s d - 3 . 3 . m s i " )  
 # D o w n l o a d   a n d   i n s t a l l   b o o s t 
 d o w n l o a d _ f i l e   " h t t p : / / b o o s t p r o . c o m / d o w n l o a d / x 6 4 / b o o s t _ 1 _ 5 1 _ s e t u p . e x e "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ b o o s t _ 1 _ 5 1 _ s e t u p . e x e " ) 
 &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ b o o s t _ 1 _ 5 1 _ s e t u p . e x e " )  
 # F F T W 
 d o w n l o a d _ f i l e   " f t p : / / f t p . f f t w . o r g / p u b / f f t w / f f t w - 3 . 3 . 2 - d l l 6 4 . z i p "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ f f t w - 3 . 3 . 2 - d l l 6 4 . z i p " ) 
 S e t - V S - E n v i r o n m e n t 
 u n z i p   ( $ d o w n l o a d _ l o c a t i o n   +   " \ f f t w - 3 . 3 . 2 - d l l 6 4 . z i p " )     " $ l i b r a r y _ l o c a t i o n \ f f t w 3 " 
 c d   " $ l i b r a r y _ l o c a t i o n \ f f t w 3 " 
 &   l i b   " / m a c h i n e : X 6 4 "   " / d e f : l i b f f t w 3 - 3 . d e f " 
 &   l i b   " / m a c h i n e : X 6 4 "   " / d e f : l i b f f t w 3 f - 3 . d e f " 
 &   l i b   " / m a c h i n e : X 6 4 "   " / d e f : l i b f f t w 3 l - 3 . d e f " 

 # M e s s a g e   r e m i n d i n g   t o   s e t   p a t h s 
 W r i t e - H o s t   " P l e a s e   e n s u r e   t h a t   p a t h s   t o   t h e   f o l l o w i n g   l o c a t i o n s   a r e   i n   y o u r   P A T H   e n v i r o n m e n t   v a r i a b l e :   " 
 W r i t e - H o s t   "         -   B o o s t   l i b r a r i e s         ( t y p i c a l l y   C : \ P r o g r a m   F i l e s \ b o o s t \ b o o s t _ 1 _ 5 1 \ l i b ) " 
 W r i t e - H o s t   "         -   C o d e   S y n t h e s i s   X S D   ( t y p i c a l l y   C : \ P r o g r a m   F i l e s   ( x 8 6 ) \ C o d e S y n t h e s i s   X S D   3 . 3 \ b i n \ ; C : \ P r o g r a m   F i l e s   ( x 8 6 ) \ C o d e S y n t h e s i s   X S D   3 . 3 \ b i n 6 4 \ ) " 
 W r i t e - H o s t   "         -   F F T W   l i b r a r i e s           ( t y p i c a l l y   C : \ M R I L i b r a r i e s \ f f t w 3 ) " 
 W r i t e - H o s t   "         -   H D F 5   l i b r a r i e s           ( t y p i c a l l y   C : \ P r o g r a m   F i l e s \ H D F   G r o u p \ H D F 5 \ 1 . 8 . 9 \ b i n ) " 
 W r i t e - H o s t   "         -   I S M R M R D                         ( t y p i c a l l y   C : \ P r o g r a m   F i l e s \ i s m r m r d \ b i n ; C : \ P r o g r a m   F i l e s \ i s m r m r d \ b i n ) "  
# N o w   d o w n l o a d   a n d   c o m p i l e   I S M R M R D 
 $ g i t _ e x e   =   " C : \ P r o g r a m   F i l e s   ( x 8 6 ) \ G i t \ b i n \ g i t . e x e "   
 $ i s m r m r d _ g i t _ u r l   =   " g i t : / / g i t . c o d e . s f . n e t / p / i s m r m r d / c o d e " 
 c d   " $ l i b r a r y _ l o c a t i o n " 
 &   $ g i t _ e x e   " c l o n e "   $ i s m r m r d _ g i t _ u r l   " i s m r m r d " 
 c d   " i s m r m r d "  
 # I f   y o u   j u s t   w a n t   t o   d o w n l o a d   t h e   z i p   f i l e 
 # d o w n l o a d _ f i l e   " h t t p : / / s o u r c e f o r g e . n e t / p r o j e c t s / i s m r m r d / f i l e s / s r c / i s m r m r d _ l a t e s t . z i p "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ i s m r m r d _ l a t e s t . z i p " ) 
 # u n z i p   ( $ d o w n l o a d _ l o c a t i o n   +   " \ i s m r m r d _ l a t e s t . z i p " )     " $ l i b r a r y _ l o c a t i o n \ i s m r m r d " 
 # c d   " $ l i b r a r y _ l o c a t i o n \ i s m r m r d " 
 N e w - I t e m   " b u i l d "   - t y p e   d i r e c t o r y 
 c d   b u i l d 
 &   c m a k e   " - G "   " V i s u a l   S t u d i o   1 0   W i n 6 4 "   " - D B O O S T _ R O O T = C : / P r o g r a m   F i l e s / b o o s t / b o o s t _ 1 _ 5 1 "   " - D X E R C E S C _ I N C L U D E _ D I R = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 / i n c l u d e / x e r c e s c "   " - D X E R C E S C _ L I B R A R I E S = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 / l i b 6 4 / v c - 1 0 . 0 / x e r c e s - c _ 3 . l i b "   " - D X S D _ D I R = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 "   " - D F F T W 3 _ I N C L U D E _ D I R = C : / M R I L i b r a r i e s / f f t w 3 "   " - D F F T W 3 F _ L I B R A R Y = C : / M R I L i b r a r i e s / f f t w 3 / l i b f f t w 3 f - 3 . l i b "   " . . / "   
 m s b u i l d   . \ I S M R M R D . s l n   / p : C o n f i g u r a t i o n = R e l e a s e #After this you should install (probably as administrator) 
 # D o w n l o a d   a n d   c o m p i l e   A C E 
 d o w n l o a d _ f i l e   " h t t p : / / d o w n l o a d . d r e . v a n d e r b i l t . e d u / p r e v i o u s _ v e r s i o n s / A C E - 6 . 1 . 4 . z i p "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ A C E - 6 . 1 . 4 . z i p " ) 
 u n z i p   ( $ d o w n l o a d _ l o c a t i o n   +   " \ A C E - 6 . 1 . 4 . z i p " )   " $ l i b r a r y _ l o c a t i o n \ " 
 c d   " $ l i b r a r y _ l o c a t i o n \ A C E _ w r a p p e r s " 
 e c h o   ' # i n c l u d e   " a c e / c o n f i g - w i n 3 2 . h " '   >   a c e \ c o n f i g . h 
 e c h o   ' # d e f i n e   A C E _ N O _ I N L I N E '   > >   a c e \ c o n f i g . h 
 m s b u i l d   . \ A C E _ w r a p p e r s _ v c 1 0 . s l n   / p : C o n f i g u r a t i o n = R e l e a s e   / p : P l a t f o r m = X 6 4  
 W r i t e - H o s t   " P l e a s e   a d d   $ l i b r a r y _ l o c a t i o n \ A C E _ w r a p p e r s \ l i b   t o   y o u r   P A T H   e n v i r o n m e n t   v a r i a b l e "  
 # P y t h o n 
 d o w n l o a d _ f i l e   " h t t p : / / w w w . p y t h o n . o r g / f t p / p y t h o n / 2 . 7 . 3 / p y t h o n - 2 . 7 . 3 . a m d 6 4 . m s i "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ p y t h o n - 2 . 7 . 3 . a m d 6 4 . m s i " ) 
 &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ p y t h o n - 2 . 7 . 3 . a m d 6 4 . m s i " )  

 W r i t e - H o s t   " P l e a s e   a d d   i n s t a l l   f o l d e r   ( e . g .   C : \ P y t h o n 2 7 )   t o   P A T H   e n v i r o n m e n t   v a r i a b l e "  W r i t e - H o s t   " A d d i t i o n a l l y   a d d   a   P Y T H O N _ R O O T   e n v i r o n m e n t   v a r i a b l e "  W r i t e - H o s t   " N o w   p l e a s e   d o w n l o a d   a n d   i n s t a l l   t h e   f o l l o w i n g   p a c k a g e s   f r o m   h t t p : / / w w w . l f d . u c i . e d u / ~ g o h l k e / p y t h o n l i b s / "  W r i t e - H o s t   "   -   n u m p y - M K L - 1 . 6 . 2 . w i n - a m d 6 4 - p y 2 . 7 "  W r i t e - H o s t   "   -   s c i p y - 0 . 1 0 . 1 . w i n - a m d 6 4 - p y 2 . 7 "  W r i t e - H o s t   "   -   l i b x m l 2 - p y t h o n - 2 . 7 . 8 . w i n - a m d 6 4 - p y 2 . 7 .  e x e "  W r i t e - H o s t   "     +   a n y   a d d i t i o n a l   p a c k a g e s   t h a t   y o u   m a y   w a n t   s u c h   a s   m a t p l o t l i b ,   i P y t h o n ,   e t c . "  
 # A C M L 
 W r i t e - H o s t   " P l e a s e   d o w n l o a d   h t t p : / / d e v e l o p e r . a m d . c o m / d o w n l o a d s / a c m l 4 . 4 . 0 - w i n 6 4 . e x e "  W r i t e - H o s t   " Y o u   h a v e   t o   o p e n   y o u r   b r o w s e r   a n d   a c k n o w l e d g e   t h e   l i c e n c e   a g r e e m e n t . . "  W r i t e - H o s t   " I n s t a l l   t h e   A C M L   l i b r a r y   u s i n g   t h e   i n s t a l l a t i o n   p a c k a g e   a n d   t h e n   m o d i f y   y o u t   P A T H   t o   i n c l u d e : "  W r i t e - H o s t   " C : \ A M D \ a c m l 4 . 4 . 0 \ w i n 6 4 \ l i b ; C : \ A M D \ a c m l 4 . 4 . 0 \ w i n 6 4 _ m p \ l i b   ( o r   w h a t e v e r   y o u r   i n s t a l l a t i o n   l o c a t i o n   w a s ) "  
 # C U D A 
 d o w n l o a d _ f i l e   " h t t p : / / d e v e l o p e r . d o w n l o a d . n v i d i a . c o m / c o m p u t e / c u d a / 4 _ 2 / r e l / t o o l k i t / c u d a t o o l k i t _ 4 . 2 . 9 _ w i n _ 6 4 . m s i "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ c u d a t o o l k i t _ 4 . 2 . 9 _ w i n _ 6 4 . m s i " )  &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ c u d a t o o l k i t _ 4 . 2 . 9 _ w i n _ 6 4 . m s i " )  d o w n l o a d _ f i l e   " h t t p : / / d e v e l o p e r . d o w n l o a d . n v i d i a . c o m / c o m p u t e / c u d a / 4 _ 2 / r e l / s d k / g p u c o m p u t i n g s d k _ 4 . 2 . 9 _ w i n _ 6 4 . e x e "   ( $ d o w n l o a d _ l o c a t i o n   +   " \ g p u c o m p u t i n g s d k _ 4 . 2 . 9 _ w i n _ 6 4 . e x e " )  &   ( $ d o w n l o a d _ l o c a t i o n   +   " \ g p u c o m p u t i n g s d k _ 4 . 2 . 9 _ w i n _ 6 4 . e x e " )  
W r i t e - H o s t   " D o w n l o a d   a n d   i n s t a l l   C U L D A   D E N S E   R 1 5   ( f r e e   e d i t i o n )   f r o m   h t t p : / / w w w . c u l a t o o l s . c o m / d o w n l o a d s / d e n s e / " 
 W r i t e - H o s t   " Y o u   w i l l   n e e d   t o   r e g i s t e r   b u t   i t   i s   f r e e " 
 W r i t e - H o s t   " I n s t a l l   t h e   p a c k a g e   a n d   a d d   t h e   b i n a r y   p a t h   t o   y o u r   P A T H   e n v i r o n m e n t   v a r i a b l e "  
 # d o w n l o a d   a n d   c o m p i l e   G A D G E T R O N 
 $ g i t _ e x e   =   " C : \ P r o g r a m   F i l e s   ( x 8 6 ) \ G i t \ b i n \ g i t . e x e "    $ g a d g e t r o n _ g i t _ u r l   =   " g i t : / / g i t . c o d e . s f . n e t / p / g a d g e t r o n / g a d g e t r o n "  c d   " $ l i b r a r y _ l o c a t i o n "  &   $ g i t _ e x e   " c l o n e "   $ g a d g e t r o n _ g i t _ u r l  c d   " g a d g e t r o n "  N e w - I t e m   " b u i l d "   - t y p e   d i r e c t o r y  c d   b u i l d  &   c m a k e   " - G "   " V i s u a l   S t u d i o   1 0   W i n 6 4 "   " - D B O O S T _ R O O T = C : / P r o g r a m   F i l e s / b o o s t / b o o s t _ 1 _ 5 1 "   " - D X E R C E S C _ I N C L U D E _ D I R = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 / i n c l u d e   / x e r c e s c "   " - D X E R C E S C _ L I B R A R I E S = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 / l i b 6 4 / v c - 1 0 . 0 / x e r c e s - c _ 3 . l i b "   " - D X S D _ D I R = C : / P r o g r a m   F i l e s   ( x 8 6 ) / C o d e S y n t h e s i s   X S D   3 . 3 "   " - D F F T W 3 _ I N C L U D E _ D I R = C : / M R I L i b r a r i e s / f f t w 3 "   " - D F F T W 3 F _ L I B R A R Y = C : / M R I L i b r a r i e s / f f t w 3 / l i b f f t w 3 f - 3 . l i b "   " - D F F T W 3 _ L I B R A R Y = C : / M R I L i b r a r i e s / f f t w 3 / l i b f f t w 3 - 3 . l i b "   " - D B L A _ V E N D O R = A C M L "   " - D C M A K E _ F o r t r a n _ C O M P I L E R _ I D = P G I "   " - D I S M R M R D _ I N C L U D E _ D I R = C : / P r o g r a m   F i l e s / I S M R M R D / i s m r m r d / i n c l u d e "   " - D I S M R M R D _ S C H E M A _ D I R = C : / P r o g r a m   F i l e s / I S M R M R D / i s m r m r d / s c h e m a "   " - D C U L A _ I N C L U D E _ D I R = C : / P r o g r a m   F i l e s / C U L A / R 1 5 / i n c l u d e "   " . . / "  S e t - V S - E n v i r o n m e n t  m s b u i l d   . \ G A D G E T R O N . s l n   / p : C o n f i g u r a t i o n = R e l e a s e  #After this you should install (probably as administrator)
 
