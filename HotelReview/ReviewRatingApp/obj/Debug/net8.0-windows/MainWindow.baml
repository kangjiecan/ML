   M S B A M L   `   `   `  ����  ReviewRatingApp     ReviewRatingApp.MainWindow   5   	   W SPresentationCore, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5   	   R NSystem.Xaml, Version=8.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	     � Title$  Review Rating App��5   	   $	��400��6#   $	��600��60   .��5      � .+�5   
   �� $��Enter Review:��6   $
��Left=�6(   $	��Top=�6C   $��	10,10,0,0q�6[   �� 5   
   -   5        $��ReviewTextBox��5      $
��Left=�5	      $	��Top=�5
      $��80��5      $	��450��5      $��	10,30,0,0q�5       �� AcceptsReturn$
 True��5       �� TextWrapping$
 Wrap=�5      �� 5   
   -   5        $��PredictButton��5      $��Predict Rating��5      $
��Left=�5      $	��Top=�5      $	��150��5      $��
10,120,0,0q�5      5      �� 5   
   -   5        $��PredictedRatingLabel��5      $��Predicted Rating: ��5      $
��Left=�5      $	��Top=�5      $��180,120,0,0q�5      �� 5"   
   $��Corrected Rating:��6   $
��Left=�5#      $	��Top=�5$      $��
10,160,0,0q�5%      �� 5&   
   -   5        $��CorrectedRatingTextBox��5&      $	��100��5'      $��25��5(      $
��Left=�5)      $	��Top=�5*      $��150,160,0,0q�5+      �� 5/   
   -   5        $��RetrainButton��5/      $��Retrain Model��50      $
��Left=�51      $	��Top=�52      $	��150��53      $��
10,203,0,0q�54      55      56      57      